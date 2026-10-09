from __future__ import annotations

import os
import subprocess
import sys
from time import time

import numpy as np
import torch

from .get_data_loader import DEFAULT_POOL_OP_KERNEL_SIZES, get_data_loader
from .trainer import (
    SynWeakNNTrainer,
    as_feature_list,
    collate_outputs,
    decode_with_decoder_features,
    feature_distillation_loss,
    set_requires_grad,
    unique_params,
)


class PaperTrainer(SynWeakNNTrainer):
    """
    Paper training schedule.

    Fixed training flow:
      1) epochs [0, warmup_epochs): fully supervised training on simulatedir.
      2) run Bayesian pseudo-label refresh once after warmup.
      3) epochs [warmup_epochs, num_epochs): joint training with
         supervised loss + pseudo-label loss + synthetic regularization loss
         + feature-distillation regularization.

    The synthetic branch is used in every joint-stage iteration.
    """

    def __init__(self, dataset, output_path: str, dataset_json: str, device: torch.device, config):
        # Parent trainer expects the legacy stage attributes to exist.
        for name, value in (("start_weak", 0), ("start_one", 0), ("trainone_finetune_scope", "all")):
            if not hasattr(config, name):
                setattr(config, name, value)
        super().__init__(dataset, output_path, dataset_json, device, config)
        self.warmup_epochs = max(0, int(getattr(config, "warmup_epochs", 100)))
        self.confidence_iter_interval = int(getattr(config, "confidence_iter_interval", 10) or 0)
        self.active_weakdir = getattr(config, "weakdir")
        self.enable_ema = bool(getattr(config, "enable_ema", True))
        self.ema_decay = float(getattr(config, "ema_decay", self.ema_decay))
        self.confidence_iter_use_ema = True
        self.did_initial_confidence_iteration = False

    def initialize(self):
        super().initialize()
        set_requires_grad(self.real_net.network.backbone, True)
        set_requires_grad(self.real_net.network.decoder, True)
        if not self.share_encoder:
            set_requires_grad(self.synth_net.network.backbone, True)
        self.current_epoch = 0
        if self.loaded_synth_pretrained:
            if not self.share_encoder:
                self.real_net.network.backbone.load_state_dict(self.synth_net.network.backbone.state_dict())
            self.print_to_log_file("[paper] initialized real/synth from synth_pretrained")
        else:
            self.print_to_log_file("[paper] no synth_pretrained provided; using model initialization")
        if self.enable_ema:
            self.init_ema_real()
        self.print_to_log_file(
            f"[paper] fixed schedule: warmup_epochs={self.warmup_epochs}, "
            f"num_epochs={self.num_epochs}, confidence_iter_interval={self.confidence_iter_interval}"
        )

    def is_supervised_warmup_epoch(self, epoch: int) -> bool:
        return int(epoch) < self.warmup_epochs

    def build_loaders(self, num_processes_da=None, pin_memory=None):
        return get_data_loader(
            self.active_weakdir,
            self.config.simulatedir,
            self.dataset_json_path,
            tuple(self.config.patch_size),
            self.config.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            deep_supervision=getattr(self.config, "deep_supervision", False),
            pool_op_kernel_sizes=getattr(self.config, "pool_op_kernel_sizes", DEFAULT_POOL_OP_KERNEL_SIZES),
            num_processes_da=getattr(self.config, "num_processes_da", None) if num_processes_da is None else num_processes_da,
            pin_memory=(self.device.type == "cuda") if pin_memory is None else pin_memory,
            intensity_channels=getattr(self.config, "intensity_channels", [0]),
            load_weak_confidence=self.confidence_weight,
            left_right_pairs_csv=getattr(self.config, "left_right_pairs_csv", None),
        )

    def train_supervised_batch(self):
        self.real_optimizer.zero_grad(set_to_none=True)
        batch = next(self.simulateloader)
        data = batch["data"].to(self.device, non_blocking=True)
        target = self.target_to_device(batch["target"])
        with self.autocast_context():
            loss = self.compute_loss(self.real_net(data), target, self.loss)
        self.backward_step(loss, self.real_optimizer, unique_params(self.real_net, self.synth_net))
        return {
            "loss": np.array(0, dtype=np.float32),
            "weakloss": np.array(0, dtype=np.float32),
            "fdloss": np.array(0, dtype=np.float32),
            "loss_finetune": loss.detach().cpu().numpy(),
        }

    def train_joint_batch(self):
        self.real_optimizer.zero_grad(set_to_none=True)

        # Synthetic regularization branch: used in every joint-stage iteration.
        synth_data, synth_target = self.synth_batch_to_torch()

        weak_batch = next(self.weakloader)
        weak_data = weak_batch["data"].to(self.device, non_blocking=True)
        weak_target = self.target_to_device(weak_batch["target"])
        confidence = weak_batch.get("confidence")
        if confidence is not None:
            if isinstance(confidence, (tuple, list)):
                confidence = [c.to(self.device, non_blocking=True) for c in confidence]
            else:
                confidence = confidence.to(self.device, non_blocking=True)

        supervised_batch = next(self.simulateloader)
        supervised_data = supervised_batch["data"].to(self.device, non_blocking=True)
        supervised_target = self.target_to_device(supervised_batch["target"])

        with self.autocast_context():
            synth_loss = self.compute_loss(self.synth_net(synth_data), synth_target, self.loss)

            if self.enable_feature_distill:
                encoder_features = self.real_net.network.backbone(weak_data)
                weak_output, real_features = decode_with_decoder_features(self.real_net, encoder_features)
            else:
                encoder_features = None
                real_features = None
                weak_output = self.real_net(weak_data)

            weak_target = self.targets_for_output(weak_target, weak_output)
            if self.confidence_weight and confidence is not None:
                weakloss = self.compute_confidence_weighted_loss(weak_output, weak_target, confidence)
            else:
                weakloss = self.compute_loss(weak_output, weak_target, self.loss)

            fdloss = torch.zeros((), device=self.device)
            if self.enable_feature_distill:
                with torch.no_grad():
                    if self.share_encoder and encoder_features is not None:
                        teacher_encoder_features = [f.detach() for f in as_feature_list(encoder_features)]
                    else:
                        teacher_encoder_features = self.synth_net.network.backbone(weak_data)
                    _, teacher_features = decode_with_decoder_features(self.synth_net, teacher_encoder_features)
                fdloss = feature_distillation_loss(real_features, teacher_features, self.feature_distill_eps)

            supervised_loss = self.compute_loss(self.real_net(supervised_data), supervised_target, self.loss)
            total_loss = synth_loss + weakloss + supervised_loss + float(self.feature_distill_weight) * fdloss

        self.backward_step(total_loss, self.real_optimizer, unique_params(self.real_net, self.synth_net))
        self.weak_batch_idx += 1
        return {
            "loss": synth_loss.detach().cpu().numpy(),
            "weakloss": weakloss.detach().cpu().numpy(),
            "fdloss": fdloss.detach().cpu().numpy(),
            "loss_finetune": supervised_loss.detach().cpu().numpy(),
        }

    def apply_confidence_to_ignore(self, target, confidence):
        # Paper schedule uses confidence as a continuous weight, not as ignore-threshold masking.
        return target

    def should_run_confidence_iteration(self, epoch: int) -> bool:
        if (self.confidence_iter_interval <= 0) or (epoch + 1 >= self.num_epochs):
            return False
        if self.is_supervised_warmup_epoch(epoch):
            return False
        return (epoch + 1 - self.warmup_epochs) % self.confidence_iter_interval == 0

    def confidence_iteration_root(self) -> str:
        return getattr(self.config, "confidence_iter_output_root", None) or os.path.join(self.output_folder, "confidence_iter")

    def confidence_iteration_checkpoint(self, epoch: int) -> str:
        return os.path.join(self.output_folder, f"checkpoint_real_ema_epoch_{epoch}.pth")

    def save_ema_checkpoint(self, epoch: int):
        if not self.enable_ema:
            raise RuntimeError("Paper training requires EMA checkpoints for Bayesian iteration.")
        if not hasattr(self, "ema_real_state") or self.ema_real_state is None:
            self.init_ema_real()
        self.save_real_ema_checkpoint(self.confidence_iteration_checkpoint(epoch))

    def run_confidence_iteration(self, epoch: int):
        confidence_root = self.confidence_iteration_root()
        os.makedirs(confidence_root, exist_ok=True)

        checkpoint = self.confidence_iteration_checkpoint(epoch)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"confidence iteration checkpoint not found: {checkpoint}")

        args_json = os.path.join(self.output_folder, "args.json")
        if not os.path.exists(args_json):
            raise FileNotFoundError(f"confidence iteration args.json not found: {args_json}")

        processed_weakdir = os.path.join(confidence_root, "weak_processed")
        iter_script = getattr(self.config, "confidence_iter_script", None)
        if not iter_script:
            iter_script = str(Path(__file__).resolve().parents[1] / "online_iteration" / "iter_online.py")

        cmd = [
            sys.executable,
            iter_script,
            "--args_json",
            args_json,
            "--checkpoint",
            checkpoint,
            "--input_csv",
            str(getattr(self.config, "confidence_iter_input_csv", "")),
            "--output_root",
            confidence_root,
            "--dataset_json",
            str(getattr(self.config, "dataset_json", self.dataset_json_path)),
            "--bayes_tissue_csv",
            str(getattr(self.config, "confidence_iter_tissue_csv", "")),
            "--target_spacing",
            *(str(float(i)) for i in getattr(self.config, "confidence_iter_target_spacing", [1, 1, 1])),
        ]
        gpu = getattr(self.config, "confidence_iter_gpu", None)
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        prior_dir = getattr(self.config, "confidence_iter_prior_dir", None)
        if prior_dir:
            cmd += ["--prior_dir", str(prior_dir)]
        if getattr(self.config, "confidence_iter_force", True):
            cmd.append("--force")
        # Always update iterative synthetic labels for the synthetic regularization teacher.
        cmd.append("--update_synth")

        self.print_to_log_file(f"[confidence_iter] start epoch={epoch}, checkpoint={checkpoint}")
        self.print_to_log_file("[confidence_iter] cmd:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        self.active_weakdir = processed_weakdir
        iter_synth_dir = os.path.join(confidence_root, "iter_synth")
        if not os.path.isdir(iter_synth_dir):
            raise FileNotFoundError(f"confidence iteration synth labels not found: {iter_synth_dir}")
        self.stop_synth_prefetcher()
        self.brain_generator = None
        self.config.synthpath = iter_synth_dir
        self.print_to_log_file(f"[confidence_iter] synthpath -> {self.config.synthpath}; reset BrainGenerator")
        self.print_to_log_file(f"[confidence_iter] active weakdir -> {self.active_weakdir}; rebuilding dataloaders")
        self.weakloader, self.simulateloader = self.build_loaders()

    def maybe_run_initial_confidence_iteration(self, epoch: int):
        if self.did_initial_confidence_iteration or self.is_supervised_warmup_epoch(epoch):
            return
        checkpoint_epoch = max(0, epoch - 1)
        self.save_ema_checkpoint(checkpoint_epoch)
        self.print_to_log_file(f"[confidence_iter] initial refresh after supervised warmup using epoch={checkpoint_epoch}")
        self.run_confidence_iteration(checkpoint_epoch)
        self.did_initial_confidence_iteration = True

    def maybe_run_confidence_iteration(self, epoch: int):
        if self.should_run_confidence_iteration(epoch):
            self.save_ema_checkpoint(epoch)
            self.run_confidence_iteration(epoch)

    def run_training(self):
        self.initialize()
        self.reset_optimizer_and_scheduler(self.initial_lr, self.num_epochs, 0)
        self.weakloader, self.simulateloader = self.build_loaders()

        self.print_to_log_file(
            "[paper] schedule: "
            f"0-{self.warmup_epochs - 1}=supervised, "
            f"{self.warmup_epochs}-{self.num_epochs - 1}=joint pseudo+synth+supervised+distill, "
            f"Bayes interval={self.confidence_iter_interval}, "
            f"confidence_weight={self.confidence_weight}, feature_distill={self.enable_feature_distill}, "
            f"feature_distill_weight={self.feature_distill_weight}, ema_decay={self.ema_decay}"
        )

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            mode = "supervised_warmup" if self.is_supervised_warmup_epoch(epoch) else "joint_pseudo_regularized"
            self.real_lr_scheduler.step(epoch - self.lr_scheduler_start_epoch)
            self.maybe_run_initial_confidence_iteration(epoch)

            if self.is_supervised_warmup_epoch(epoch):
                self.stop_synth_prefetcher()
            else:
                self.start_synth_prefetcher()

            self.real_net.train()
            self.synth_net.train()
            set_requires_grad(self.real_net.network.backbone, True)
            set_requires_grad(self.real_net.network.decoder, True)
            if not self.share_encoder:
                set_requires_grad(self.synth_net.network.backbone, True)
            self.assert_shared_decoder()

            outputs = []
            epoch_start = time()
            for _ in range(self.num_iterations_per_epoch):
                try:
                    out = self.train_supervised_batch() if self.is_supervised_warmup_epoch(epoch) else self.train_joint_batch()
                    if self.enable_ema:
                        self.update_ema_real()
                    outputs.append(out)
                except RuntimeError as e:
                    msg = str(e)
                    if "background workers" not in msg:
                        raise
                    self.print_to_log_file(
                        "[dataloader] background workers failed; falling back to single-process loader. Error:", msg
                    )
                    self.weakloader, self.simulateloader = self.build_loaders(num_processes_da=0, pin_memory=False)
                    out = self.train_supervised_batch() if self.is_supervised_warmup_epoch(epoch) else self.train_joint_batch()
                    if self.enable_ema:
                        self.update_ema_real()
                    outputs.append(out)

            logs = collate_outputs(outputs)
            synth_loss = float(np.mean(logs["loss"]))
            weakloss = float(np.mean(logs["weakloss"]))
            fdloss = float(np.mean(logs.get("fdloss", np.array([0], dtype=np.float32))))
            supervised_loss = float(np.mean(logs["loss_finetune"]))
            self.logger.log("train_losses_synth", synth_loss, epoch)
            self.logger.log("train_losses_weak", weakloss, epoch)
            self.logger.log("train_losses_fd", fdloss, epoch)
            self.logger.log("train_losses_finetune", supervised_loss, epoch)
            self.print_to_log_file(
                f"Epoch {epoch} [{mode}]: synth={synth_loss:.4f}, weak={weakloss:.4f}, "
                f"fd={fdloss:.4f}, supervised={supervised_loss:.4f}, time={time() - epoch_start:.2f}s"
            )
            self.logger.plot_progress_png(self.output_folder)

            if ((epoch + 1) % self.save_every == 0) or self.should_run_confidence_iteration(epoch):
                self.save_ema_checkpoint(epoch)

            if not self.is_supervised_warmup_epoch(epoch):
                self.maybe_run_confidence_iteration(epoch)

        self.save_ema_checkpoint(self.num_epochs - 1)
        self.stop_synth_prefetcher()
