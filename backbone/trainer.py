from __future__ import annotations

import csv
import os
import sys
import queue
import threading
from contextlib import nullcontext
from datetime import datetime
from time import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import GradScaler, autocast

from .get_data_loader import DEFAULT_POOL_OP_KERNEL_SIZES, get_data_loader, load_dataset_json
from .label_manager import build_label_manager
from .losses import build_confidence_weighted_loss, build_nnunet_loss, deep_supervision_scales
from .paths import add_project_paths
from .transforms import DEFAULT_LEFT_RIGHT_LABEL_PAIRS, _normalize_lr_structure_name, infer_left_right_pairs_from_csv

add_project_paths()

from .getmodel import build_model  # noqa: E402


def get_brain_generator_cls():
    from .synthimg.brain_generator import BrainGenerator

    return BrainGenerator


DEFAULT_SYNTH_GENERATION_LABELS = np.array(
    [
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37,
    ],
    dtype=np.int32,
)
DEFAULT_SYNTH_N_NEUTRAL_LABELS = 6


def synth_generation_labels_from_table(csv_path: str | None):
    if csv_path is None or str(csv_path).strip() == "":
        return DEFAULT_SYNTH_GENERATION_LABELS.copy(), DEFAULT_SYNTH_N_NEUTRAL_LABELS

    neutral_labels = []
    left_labels = []
    right_labels = []
    seen = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "Structure" not in reader.fieldnames or "GGBond" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain Structure and GGBond columns")
        for row in reader:
            label_text = str(row.get("GGBond", "")).strip()
            if label_text == "":
                continue
            try:
                label = int(float(label_text))
            except ValueError:
                continue
            if label in seen:
                continue
            seen.add(label)
            side, _ = _normalize_lr_structure_name(row.get("Structure", ""))
            if side == "left":
                left_labels.append(label)
            elif side == "right":
                right_labels.append(label)
            else:
                neutral_labels.append(label)

    if not neutral_labels:
        raise RuntimeError(f"No neutral labels found in {csv_path}")
    if not left_labels or not right_labels:
        raise RuntimeError(f"Could not infer both left and right labels from {csv_path}")

    neutral_labels = sorted(neutral_labels)
    left_labels = sorted(left_labels)
    right_labels = sorted(right_labels)
    generation_labels = np.asarray(neutral_labels + left_labels + right_labels, dtype=np.int32)
    return generation_labels, len(neutral_labels)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def unique_params(*modules):
    seen = set()
    params = []
    for module in modules:
        for p in module.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("network_weights", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def set_deep_supervision_enabled(model, enabled: bool):
    changed = False
    if hasattr(model, "deep_supervision"):
        model.deep_supervision = enabled
        changed = True
    net = getattr(model, "network", None)
    if net is not None and hasattr(net, "deep_supervision"):
        net.deep_supervision = enabled
        changed = True
    decoder = getattr(net, "decoder", None) if net is not None else None
    if decoder is not None and hasattr(decoder, "deep_supervision"):
        decoder.deep_supervision = enabled
        changed = True
    if decoder is not None and hasattr(decoder, "do_ds"):
        if enabled and not hasattr(decoder, "out_1"):
            raise RuntimeError("This model was built without deep-supervision heads. Please use a model that supports enabling deep supervision after construction.")
        decoder.do_ds = enabled
        changed = True
    if enabled and not changed:
        raise RuntimeError(f"Could not enable deep supervision for {type(model).__name__}")



def as_feature_list(features):
    return list(features) if isinstance(features, (tuple, list)) else [features]


def interpolate_feature_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[2:] == target.shape[2:]:
        return source
    spatial_dims = source.ndim - 2
    if spatial_dims == 3:
        return F.interpolate(source, size=target.shape[2:], mode="trilinear", align_corners=False)
    if spatial_dims == 2:
        return F.interpolate(source, size=target.shape[2:], mode="bilinear", align_corners=False)
    return F.interpolate(source, size=target.shape[2:], mode="nearest")


def feature_distillation_loss(real_features, teacher_features, eps: float = 1e-8):
    """Multi-scale cosine feature regularization: mean(1 - cosine(student, teacher))."""
    real_features = as_feature_list(real_features)
    teacher_features = [t.detach() for t in as_feature_list(teacher_features)]
    layer_count = min(len(real_features), len(teacher_features))
    if layer_count == 0:
        device = real_features[0].device if real_features else "cpu"
        return torch.zeros((), device=device)
    total = torch.zeros((), device=real_features[0].device)
    used_layers = 0
    for real_feat, teacher_feat in zip(real_features[:layer_count], teacher_features[:layer_count]):
        if real_feat.shape[1] != teacher_feat.shape[1]:
            continue
        real_feat = interpolate_feature_like(real_feat, teacher_feat).float()
        teacher_feat = teacher_feat.detach().float()
        layer_loss = 1.0 - F.cosine_similarity(real_feat, teacher_feat, dim=1, eps=eps)
        total = total + layer_loss.mean()
        used_layers += 1
    if used_layers == 0:
        return torch.zeros((), device=real_features[0].device)
    return total / float(used_layers)

def local_affinity_distribution(feature: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if feature.ndim != 5:
        raise ValueError(f"local affinity distillation expects 5D B,C,X,Y,Z features, got {tuple(feature.shape)}")
    if min(feature.shape[2:]) < 3:
        raise ValueError(f"feature spatial shape is too small for 3x3x3 affinity: {tuple(feature.shape[2:])}")
    normed = F.normalize(feature.float(), p=2, dim=1, eps=eps)
    center = normed[:, :, 1:-1, 1:-1, 1:-1]
    affinities = []
    for dx in (-1, 0, 1):
        xs = slice(1 + dx, feature.shape[2] - 1 + dx)
        for dy in (-1, 0, 1):
            ys = slice(1 + dy, feature.shape[3] - 1 + dy)
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                zs = slice(1 + dz, feature.shape[4] - 1 + dz)
                neighbor = normed[:, :, xs, ys, zs]
                affinities.append(torch.sum(center * neighbor, dim=1))
    affinity = torch.stack(affinities, dim=1)
    return torch.softmax(affinity, dim=1)


def affinity_distillation_loss(real_features, teacher_features, eps: float = 1e-8):
    real_features = as_feature_list(real_features)
    teacher_features = [t.detach() for t in as_feature_list(teacher_features)]
    layer_count = min(len(real_features), len(teacher_features))
    if layer_count == 0:
        device = real_features[0].device if real_features else "cpu"
        return torch.zeros((), device=device)
    total = torch.zeros((), device=real_features[0].device)
    used_layers = 0
    for idx, (real_feat, teacher_feat) in enumerate(zip(real_features[:layer_count], teacher_features[:layer_count])):
        alpha = 1.0 if layer_count == 1 else idx / (layer_count - 1)
        if float(alpha) == 0.0:
            continue
        if real_feat.shape[1] != teacher_feat.shape[1]:
            continue
        if real_feat.ndim != 5 or teacher_feat.ndim != 5:
            continue
        real_feat = interpolate_feature_like(real_feat, teacher_feat)
        if min(real_feat.shape[2:]) < 3 or min(teacher_feat.shape[2:]) < 3:
            continue
        student_affinity = local_affinity_distribution(real_feat, eps)
        with torch.no_grad():
            teacher_affinity = local_affinity_distribution(teacher_feat, eps).detach()
        layer_loss = torch.sum(
            teacher_affinity.clamp_min(eps)
            * (torch.log(teacher_affinity.clamp_min(eps)) - torch.log(student_affinity.clamp_min(eps))),
            dim=1,
        ).mean()
        total = total + float(alpha) * layer_loss
        used_layers += 1
    if used_layers == 0:
        return torch.zeros((), device=real_features[0].device)
    return total



def decoder_feature_modules(model):
    """Select decoder modules whose outputs serve as multi-scale decoding features."""
    net = getattr(model, "network", None)
    decoder = getattr(net, "decoder", None) if net is not None else None
    if decoder is None:
        return []
    modules = []
    if hasattr(decoder, "fusion_conv"):
        modules.append(decoder.fusion_conv)
    if hasattr(decoder, "decoders"):
        modules.extend(list(decoder.decoders))
    if not modules and hasattr(decoder, "seg_heads"):
        modules.extend(list(decoder.seg_heads))
    return modules


def decode_with_decoder_features(model, features):
    """Decode encoder features while collecting decoder feature maps by forward hooks."""
    captured = []
    handles = []
    for module in decoder_feature_modules(model):
        handles.append(module.register_forward_hook(lambda _m, _inp, out: captured.append(out)))
    try:
        output = decode_features_like_model(model, features)
    finally:
        for handle in handles:
            handle.remove()
    captured = [x for x in captured if torch.is_tensor(x)]
    return output, captured


def apply_model_output_policy(model, seg_output):
    """Match the wrapper model's deep-supervision output policy after manual decoding."""
    if getattr(model, "deep_supervision", False):
        return seg_output if isinstance(seg_output, (tuple, list)) else [seg_output]
    return seg_output[0] if isinstance(seg_output, (tuple, list)) else seg_output


def decode_features_like_model(model, features):
    features_for_decoder = list(features) if isinstance(features, list) else list(features) if isinstance(features, tuple) else features
    seg_output = model.network.decoder(features_for_decoder)
    return apply_model_output_policy(model, seg_output)


class SynthBatchPrefetcher:
    def __init__(self, make_batch_fn, maxsize: int = 2):
        self.make_batch_fn = make_batch_fn
        self.queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.make_batch_fn()
            except BaseException as e:
                item = e
            while not self.stop_event.is_set():
                try:
                    self.queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def next(self):
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self.stop_event.set()


class PolyLRScheduler:
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent

    def step(self, current_step: int):
        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


def collate_outputs(outputs):
    keys = outputs[0].keys()
    return {k: np.vstack([o[k] for o in outputs]) for k in keys}


class SimpleLogger:
    def __init__(self):
        self.history = {}

    def log(self, key, value, epoch):
        self.history.setdefault(key, []).append(float(value))

    def plot_progress_png(self, output_folder: str):
        if not self.history:
            return
        os.makedirs(output_folder, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 7))
        labels = {
            "train_losses_synth": "synth",
            "train_losses_weak": "weak",
            "train_losses_fd": "fd",
            "train_losses_finetune": "finetune",
        }
        for key, label in labels.items():
            values = self.history.get(key, [])
            if len(values) > 0:
                ax.plot(np.arange(len(values)), values, label=label, linewidth=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, "progress.png"))
        plt.close(fig)


class SynWeakNNTrainer:
    def __init__(self, dataset, output_path: str, dataset_json: str, device: torch.device, config):
        self.device = torch.device("cuda", 0) if device.type == "cuda" else device
        print(f"Using device: {self.device}")
        self.output_folder = output_path
        self.dataset = dataset
        self.dataset_json_path = dataset_json
        self.dataset_info = load_dataset_json(dataset_json)
        self.label_manager = build_label_manager(self.dataset_info)
        self.config = config

        self.initial_lr = getattr(config, "initial_lr", 1e-3)
        self.trainone_lr = getattr(config, "trainone_lr", 1e-3)
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = getattr(config, "num_iterations_per_epoch", 250)
        self.num_epochs = config.num_epochs
        self.current_epoch = 0
        self.start_weak = config.start_weak
        self.start_one = config.start_one
        self.save_every = getattr(config, "save_every", 5)
        self.ema_decay = 0.995
        self.trainone_finetune_scope = getattr(config, "trainone_finetune_scope", "all")
        if self.trainone_finetune_scope not in ("all", "decoder"):
            raise ValueError("trainone_finetune_scope must be either 'all' or 'decoder'")
        self.confidence_threshold = getattr(config, "confidence_threshold", 0.5)
        self.confidence_ignore_label = getattr(config, "confidence_ignore_label", 1000)
        self.enable_confidence_ignore = getattr(config, "enable_confidence_ignore", True)
        self.confidence_weight = getattr(config, "confidence_weight", False)
        self.confidence_power = float(getattr(config, "confidence_power", 1.0))
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be > 0")
        self.share_encoder = bool(getattr(config, "share_encoder", False))
        self.deep_supervision = getattr(config, "deep_supervision", False)
        self.ds_weights = None
        self.enable_feature_distill = getattr(config, "enable_feature_distill", False)
        self.enable_affinity = getattr(config, "enable_affinity", False)
        self.feature_distill_weight = getattr(config, "feature_distill_weight", 0.1)
        self.feature_distill_eps = getattr(config, "feature_distill_eps", 1e-8)
        self.weak_synth_interval = getattr(config, "weak_synth_interval", 10)
        self.weak_batch_idx = 0
        self.enable_weak_synth_training = getattr(config, "enable_weak_synth_training", True)
        self.enable_synth_prefetch = getattr(config, "enable_synth_prefetch", True)
        self.synth_prefetch_size = getattr(config, "synth_prefetch_size", 2)
        self.fold_right_to_left = getattr(config, "fold_right_to_left", False)
        self.left_right_label_pairs = infer_left_right_pairs_from_csv(getattr(config, "left_right_pairs_csv", None))
        if self.left_right_label_pairs is None:
            self.left_right_label_pairs = DEFAULT_LEFT_RIGHT_LABEL_PAIRS
        self.synth_pretrained = getattr(config, "synth_pretrained", None)
        self.skip_synth_warmup_when_pretrained = getattr(config, "skip_synth_warmup_when_pretrained", True)
        self.loaded_synth_pretrained = False

        self.synth_net = None
        self.real_net = None
        self.real_optimizer = None
        self.real_lr_scheduler = None
        self.lr_scheduler_start_epoch = 0
        self.loss = None
        self.weak_loss = None
        self.grad_scaler = GradScaler("cuda") if self.device.type == "cuda" else None
        self.logger = SimpleLogger()
        self.ema_decoder_state = None
        self.brain_generator = None
        self.synth_prefetcher = None
        self.was_initialized = False

        timestamp = datetime.now()
        self.log_file = os.path.join(
            self.output_folder,
            "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt"
            % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute, timestamp.second),
        )

    def print_to_log_file(self, *args, also_print_to_console=True):
        os.makedirs(self.output_folder, exist_ok=True)
        text = f"{datetime.fromtimestamp(time())}: " + " ".join(str(a) for a in args)
        with open(self.log_file, "a+") as f:
            f.write(text + "\n")
        if also_print_to_console:
            print(text)

    def initialize(self):
        if self.was_initialized:
            return
        self.real_net = build_model(self.config.modelname, self.config.in_channels, self.config.num_classes)
        self.synth_net = build_model(self.config.modelname, self.config.in_channels, self.config.num_classes)
        if self.share_encoder:
            self.synth_net.network.backbone = self.real_net.network.backbone
        else:
            self.synth_net.network.decoder = self.real_net.network.decoder
        if self.deep_supervision:
            set_deep_supervision_enabled(self.real_net, True)
            set_deep_supervision_enabled(self.synth_net, True)
        self.real_net.to(self.device)
        self.synth_net.to(self.device)

        pool_op_kernel_sizes = getattr(self.config, "pool_op_kernel_sizes", DEFAULT_POOL_OP_KERNEL_SIZES)
        self.loss = build_nnunet_loss(
            self.label_manager,
            batch_dice=self.config.batch_dice,
            deep_supervision=False,
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            is_ddp=False,
            compile_dice=getattr(self.config, "compile_dice", False),
            ignore_label_override=1000
        )
        self.weak_loss = build_nnunet_loss(
            self.label_manager,
            batch_dice=self.config.batch_dice,
            deep_supervision=False,
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            is_ddp=False,
            compile_dice=getattr(self.config, "compile_dice", False),
            ignore_label_override=self.confidence_ignore_label,
        )
        self.weighted_weak_loss = build_confidence_weighted_loss(
            self.label_manager,
            batch_dice=self.config.batch_dice,
            is_ddp=False,
            confidence_power=self.confidence_power,
        ) if self.confidence_weight else None
        if self.deep_supervision:
            scales = deep_supervision_scales(pool_op_kernel_sizes, True)
            weights = np.array([1 / (2 ** i) for i in range(len(scales))], dtype=np.float32)
            weights[-1] = 0
            self.ds_weights = weights / weights.sum()
        self.load_synth_pretrained()
        os.makedirs(self.output_folder, exist_ok=True)
        self.was_initialized = True

    def assert_shared_decoder(self):
        if self.share_encoder:
            assert self.synth_net.network.backbone is self.real_net.network.backbone, "encoder/backbone module is not shared"
        else:
            assert self.synth_net.network.decoder is self.real_net.network.decoder, "decoder module is not shared"

    def need_synth_for_epoch(self, epoch: int) -> bool:
        return epoch < self.start_weak or (
            epoch < self.start_one and self.enable_weak_synth_training and int(self.weak_synth_interval) > 0
        )

    def get_synth_generator(self):
        if self.brain_generator is not None:
            return
        generation_labels, n_neutral_labels = synth_generation_labels_from_table(
            getattr(self.config, "left_right_pairs_csv", None)
        )
        self.print_to_log_file(
            f"[synth] generation_labels={generation_labels.tolist()}, n_neutral_labels={n_neutral_labels}",
            also_print_to_console=True,
        )
        BrainGenerator = get_brain_generator_cls()
        self.brain_generator = BrainGenerator(
            labels_dir=self.config.synthpath,
            output_shape=self.config.patch_size,
            batchsize=self.config.batch_size,
            generation_labels=generation_labels,
            n_neutral_labels=n_neutral_labels,
            output_labels=generation_labels.copy(),
            flipping=True,
            scaling_bounds=0.1,
            rotation_bounds=False,
            shearing_bounds=False,
        )

    def synth_batch_numpy(self):
        self.get_synth_generator()
        return self.brain_generator.generate_brain()

    def start_synth_prefetcher(self):
        if not self.enable_synth_prefetch or self.synth_prefetcher is not None:
            return
        self.get_synth_generator()
        self.synth_prefetcher = SynthBatchPrefetcher(self.synth_batch_numpy, self.synth_prefetch_size)
        self.print_to_log_file(f"[synth_prefetch] started with queue size={self.synth_prefetch_size}")

    def stop_synth_prefetcher(self):
        if self.synth_prefetcher is not None:
            self.synth_prefetcher.close()
            self.synth_prefetcher = None
            self.print_to_log_file("[synth_prefetch] stopped")

    def synth_batch_to_torch(self):
        if self.synth_prefetcher is not None:
            data, target = self.synth_prefetcher.next()
        else:
            data, target = self.synth_batch_numpy()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
        if (36 in np.unique(target)) or (37 in np.unique(target)):
            raise ValueError("Synth target should not contain 36/37 after SynthSeg mapping")
        data = torch.from_numpy(data).unsqueeze(1).to(self.device, non_blocking=True)
        target = torch.from_numpy(target).unsqueeze(1).long().to(self.device, non_blocking=True)
        target = self.fold_target_right_to_left(target)
        return data, target

    def autocast_context(self):
        return autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()

    def reset_optimizer_and_scheduler(self, initial_lr: float, max_steps: int, start_epoch: int):
        max_steps = max(1, int(max_steps))
        self.real_optimizer = torch.optim.SGD(
            unique_params(self.real_net, self.synth_net),
            initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        self.real_lr_scheduler = PolyLRScheduler(self.real_optimizer, initial_lr, max_steps)
        self.lr_scheduler_start_epoch = int(start_epoch)
        self.print_to_log_file(
            f"[optimizer] reset SGD PolyLR: lr={initial_lr}, max_steps={max_steps}, start_epoch={start_epoch}"
        )

    def load_synth_pretrained(self):
        if not self.synth_pretrained:
            return
        checkpoint = torch.load(self.synth_pretrained, map_location=self.device)
        state = unwrap_checkpoint_state(checkpoint)

        current_state = self.synth_net.state_dict()
        loadable_state = {}
        skipped_shape = {}
        unexpected_keys = []
        for k, v in state.items():
            if k not in current_state:
                unexpected_keys.append(k)
                print(f'skip {k}')
                continue
            if current_state[k].shape != v.shape:
                skipped_shape[k] = {
                    "checkpoint": tuple(v.shape),
                    "current": tuple(current_state[k].shape),
                }
                print(f'skip {k}')
                continue
            loadable_state[k] = v   

        missing, unexpected = self.synth_net.load_state_dict(state, strict=False)
        real_missing, real_unexpected = [], []
        if self.share_encoder:
            real_missing, real_unexpected = self.real_net.load_state_dict(state, strict=False)

        self.loaded_synth_pretrained = True
        msg = f"[synth_pretrained] loaded {self.synth_pretrained}; missing={len(missing)}, unexpected={len(unexpected)}"
        if self.share_encoder:
            msg += f"; real_init_missing={len(real_missing)}, real_init_unexpected={len(real_unexpected)}"
        self.print_to_log_file(msg)

    def fold_target_right_to_left(self, target):
        if not self.fold_right_to_left:
            return target
        if isinstance(target, (tuple, list)):
            return [self.fold_target_right_to_left(t) for t in target]
        out = target.clone()
        for left_label, right_label in self.left_right_label_pairs:
            out[target == int(right_label)] = int(left_label)
        return out

    def target_to_device(self, target):
        if isinstance(target, (tuple, list)):
            return [self.target_to_device(t) for t in target]
        target = target.long().to(self.device, non_blocking=True)
        return self.fold_target_right_to_left(target)

    def downsample_target_like_output(self, target, output):
        if target.shape[2:] == output.shape[2:]:
            return target
        return F.interpolate(target.float(), size=output.shape[2:], mode="nearest").long()

    def targets_for_output(self, target, output):
        if isinstance(output, (tuple, list)):
            if isinstance(target, (tuple, list)):
                return [self.downsample_target_like_output(t, o) for t, o in zip(target, output)]
            return [self.downsample_target_like_output(target, o) for o in output]
        if isinstance(target, (tuple, list)):
            target = target[0]
        return self.downsample_target_like_output(target, output)

    def apply_confidence_to_ignore(self, target, confidence):
        if (not self.enable_confidence_ignore) or confidence is None:
            return target
        if isinstance(target, (tuple, list)):
            if isinstance(confidence, (tuple, list)):
                return [self.apply_confidence_to_ignore(t, c) for t, c in zip(target, confidence)]
            return [self.apply_confidence_to_ignore(t, confidence) for t in target]
        if isinstance(confidence, (tuple, list)):
            confidence = confidence[0]
        conf = confidence.to(device=target.device, dtype=torch.float32)
        if conf.shape[2:] != target.shape[2:]:
            conf = F.interpolate(conf, size=target.shape[2:], mode="nearest")
        out = target.clone()
        out[conf < float(self.confidence_threshold)] = int(self.confidence_ignore_label)
        return out


    def confidence_for_output(self, confidence, output):
        if confidence is None:
            return None
        if isinstance(output, (tuple, list)):
            if isinstance(confidence, (tuple, list)):
                return [self.confidence_for_output(c, o) for c, o in zip(confidence, output)]
            return [self.confidence_for_output(confidence, o) for o in output]
        if isinstance(confidence, (tuple, list)):
            confidence = confidence[0]
        conf = confidence.to(device=output.device, dtype=torch.float32)
        if conf.shape[2:] != output.shape[2:]:
            conf = F.interpolate(conf, size=output.shape[2:], mode="nearest")
        return conf

    def compute_confidence_weighted_loss(self, output, target, confidence):
        if isinstance(output, (tuple, list)):
            targets = self.targets_for_output(target, output)
            confidences = self.confidence_for_output(confidence, output)
            if self.ds_weights is not None and len(self.ds_weights) >= len(output):
                weights = self.ds_weights
            elif len(output) == 1:
                weights = np.array([1.0], dtype=np.float32)
            else:
                weights = np.array([1 / (2 ** i) for i in range(len(output))], dtype=np.float32)
                weights[-1] = 0
                weights = weights / weights.sum()
            loss = torch.zeros((), device=output[0].device)
            for i, (out, tgt, conf) in enumerate(zip(output, targets, confidences)):
                if weights[i] != 0:
                    loss = loss + float(weights[i]) * self.weighted_weak_loss(out, tgt, conf)
            return loss
        target = self.targets_for_output(target, output)
        confidence = self.confidence_for_output(confidence, output)
        return self.weighted_weak_loss(output, target, confidence)

    def compute_loss(self, output, target, loss_fn):
        if isinstance(output, (tuple, list)):
            targets = self.targets_for_output(target, output)
            if self.ds_weights is not None and len(self.ds_weights) >= len(output):
                weights = self.ds_weights
            elif len(output) == 1:
                weights = np.array([1.0], dtype=np.float32)
            else:
                weights = np.array([1 / (2 ** i) for i in range(len(output))], dtype=np.float32)
                weights[-1] = 0
                weights = weights / weights.sum()
            loss = torch.zeros((), device=output[0].device)
            for i, (out, tgt) in enumerate(zip(output, targets)):
                if weights[i] != 0:
                    loss = loss + float(weights[i]) * loss_fn(out, tgt)
            return loss
        target = self.targets_for_output(target, output)
        return loss_fn(output, target)

    def backward_step(self, loss, optimizer, params):
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 12)
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 12)
            optimizer.step()

    def train_synth_batch(self):
        data, target = self.synth_batch_to_torch()
        self.real_optimizer.zero_grad(set_to_none=True)
        with self.autocast_context():
            loss = self.compute_loss(self.synth_net(data), target, self.loss)
        self.backward_step(loss, self.real_optimizer, self.synth_net.parameters())
        loss_log = loss.detach().cpu()
        return {"loss": loss_log.numpy(), "weakloss": np.array(0, dtype=np.float32), "fdloss": np.array(0, dtype=np.float32), "loss_finetune": np.array(0, dtype=np.float32)}

    def train_weak_batch(self):
        batch_start = time()
        self.real_optimizer.zero_grad(set_to_none=True)
        synth_loss = torch.zeros((), device=self.device)
        do_synth = (
            self.enable_weak_synth_training
            and int(self.weak_synth_interval) > 0
            and self.weak_batch_idx % int(self.weak_synth_interval) == 0
        )
        if do_synth:
            data, target = self.synth_batch_to_torch()
            with self.autocast_context():
                synth_output = self.synth_net(data)
                synth_loss = self.compute_loss(synth_output, target, self.loss)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(synth_loss).backward()
            else:
                synth_loss.backward()

        weak_batch = next(self.weakloader)
        weak_data = weak_batch["data"].to(self.device, non_blocking=True)
        weak_target = self.target_to_device(weak_batch["target"])
        confidence = weak_batch.get("confidence")

        if confidence is not None:
            if isinstance(confidence, (tuple, list)):
                confidence = [c.to(self.device, non_blocking=True) for c in confidence]
            else:
                confidence = confidence.to(self.device, non_blocking=True)
        with self.autocast_context():
            if self.enable_feature_distill:
                encoder_features = self.real_net.network.backbone(weak_data)
                weak_output, real_features = decode_with_decoder_features(self.real_net, encoder_features)
            else:
                real_features = None
                weak_output = self.real_net(weak_data)

            weak_target = self.targets_for_output(weak_target, weak_output)
            use_confidence_weight = self.confidence_weight and confidence is not None
            use_confidence_ignore = (not use_confidence_weight) and self.enable_confidence_ignore and confidence is not None
            if use_confidence_weight:
                weakloss = self.compute_confidence_weighted_loss(weak_output, weak_target, confidence)
            else:
                if use_confidence_ignore:
                    weak_target = self.apply_confidence_to_ignore(weak_target, confidence)
                weakloss = self.compute_loss(weak_output, weak_target, self.weak_loss if use_confidence_ignore else self.loss)
            fdloss = torch.zeros((), device=self.device)
            if self.enable_feature_distill:
                with torch.no_grad():
                    if self.share_encoder and encoder_features is not None:
                        teacher_encoder_features = [f.detach() for f in as_feature_list(encoder_features)]
                    else:
                        teacher_encoder_features = self.synth_net.network.backbone(weak_data)
                    _, teacher_features = decode_with_decoder_features(self.synth_net, teacher_encoder_features)
                if self.enable_affinity:
                    fdloss = affinity_distillation_loss(real_features, teacher_features, self.feature_distill_eps)
                else:
                    fdloss = feature_distillation_loss(real_features, teacher_features, self.feature_distill_eps)
            total_weak_loss = weakloss + float(self.feature_distill_weight) * fdloss
        self.backward_step(total_weak_loss, self.real_optimizer, unique_params(self.real_net, self.synth_net))
        self.weak_batch_idx += 1
        loss_log = synth_loss.detach().cpu()
        weakloss_log = weakloss.detach().cpu()
        fdloss_log = fdloss.detach().cpu()
        print(f"This batch took {time() - batch_start} seconds")
        return {
            "loss": loss_log.numpy(),
            "weakloss": weakloss_log.numpy(),
            "fdloss": fdloss_log.numpy(),
            "loss_finetune": np.array(0, dtype=np.float32),
        }

    def init_ema_real(self):
        self.ema_real_state = {}
        for k, v in self.real_net.state_dict().items():
            if torch.is_tensor(v):
                self.ema_real_state[k] = v.detach().float().cpu().clone() if v.dtype.is_floating_point else v.detach().cpu().clone()
            else:
                self.ema_real_state[k] = v
        self.print_to_log_file(f"[EMA] Initialized real_net EMA with decay={self.ema_decay}")

    def update_ema_real(self):
        if not hasattr(self, "ema_real_state") or self.ema_real_state is None:
            return
        with torch.no_grad():
            for k, v in self.real_net.state_dict().items():
                if not torch.is_tensor(v):
                    self.ema_real_state[k] = v
                    continue
                v_cpu = v.detach().cpu()
                if v.dtype.is_floating_point:
                    v_cpu = v_cpu.float()
                    self.ema_real_state[k].mul_(self.ema_decay).add_(v_cpu, alpha=1.0 - self.ema_decay)
                else:
                    self.ema_real_state[k] = v_cpu.clone()

    def save_real_ema_checkpoint(self, filename: str):
        if not hasattr(self, "ema_real_state") or self.ema_real_state is None:
            return
        ema_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in self.ema_real_state.items()}
        torch.save({"network_weights": ema_state}, filename)
        # torch.save({"network_weights": ema_state, "ema_decay": self.ema_decay, "ema_scope": "real_net"}, filename)

    def init_trainone_ema(self):
        if self.trainone_finetune_scope == "all":
            self.init_ema_real()
        else:
            self.init_ema_decoder()

    def update_trainone_ema(self):
        if self.trainone_finetune_scope == "all":
            self.update_ema_real()
        else:
            self.update_ema_decoder()

    def save_trainone_ema_checkpoint(self, filename: str):
        if self.trainone_finetune_scope == "all":
            self.save_real_ema_checkpoint(filename)
        else:
            self.save_real_ema_decoder_checkpoint(filename)

    def init_ema_decoder(self):
        self.ema_decoder_state = {}
        for k, v in self.real_net.network.decoder.state_dict().items():
            self.ema_decoder_state[k] = v.detach().float().cpu().clone() if v.dtype.is_floating_point else v.detach().cpu().clone()
        self.print_to_log_file(f"[EMA] Initialized decoder EMA with decay={self.ema_decay}")

    def update_ema_decoder(self):
        if self.ema_decoder_state is None:
            return
        with torch.no_grad():
            for k, v in self.real_net.network.decoder.state_dict().items():
                v_cpu = v.detach().cpu()
                if v.dtype.is_floating_point:
                    v_cpu = v_cpu.float()
                    self.ema_decoder_state[k].mul_(self.ema_decay).add_(v_cpu, alpha=1.0 - self.ema_decay)
                else:
                    self.ema_decoder_state[k] = v_cpu.clone()

    def train_one_batch(self):
        batch = next(self.simulateloader)
        data = batch["data"].to(self.device, non_blocking=True)
        target = self.target_to_device(batch["target"])
        self.real_optimizer.zero_grad(set_to_none=True)
        with self.autocast_context():
            loss = self.compute_loss(self.real_net(data), target, self.loss)
        self.backward_step(loss, self.real_optimizer, self.real_net.parameters())
        self.update_trainone_ema()
        loss_log = loss.detach().cpu()
        return {"loss": np.array(0, dtype=np.float32), "weakloss": np.array(0, dtype=np.float32), "fdloss": np.array(0, dtype=np.float32), "loss_finetune": loss_log.numpy()}

    def save_real_checkpoint(self, filename: str):
        torch.save({"network_weights": self.real_net.state_dict()}, filename)

    def save_synth_checkpoint(self, filename: str):
        torch.save({"network_weights": self.synth_net.state_dict()}, filename)

    def save_real_ema_decoder_checkpoint(self, filename: str):
        if self.ema_decoder_state is None:
            return
        full_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in self.real_net.state_dict().items()}
        for k, v in self.ema_decoder_state.items():
            full_key = "network.decoder." + k
            if full_key in full_state:
                full_state[full_key] = v.detach().cpu().clone()
        torch.save({"network_weights": full_state, "ema_decay": self.ema_decay, "ema_scope": "network.decoder"}, filename)

    def run_training(self):
        self.initialize()
        if self.loaded_synth_pretrained and self.skip_synth_warmup_when_pretrained and self.current_epoch < self.start_weak:
            self.current_epoch = self.start_weak
            self.print_to_log_file(f"[synth_pretrained] skip synth warmup and start from epoch {self.current_epoch}")
        synth_weak_max_steps = self.start_one - self.current_epoch
        self.reset_optimizer_and_scheduler(self.initial_lr, synth_weak_max_steps, self.current_epoch)
        self.weakloader, self.simulateloader = get_data_loader(
            self.config.weakdir,
            self.config.simulatedir,
            self.dataset_json_path,
            tuple(self.config.patch_size),
            self.config.batch_size,
            oversample_foreground_percent=self.oversample_foreground_percent,
            deep_supervision=getattr(self.config, "deep_supervision", False),
            pool_op_kernel_sizes=getattr(self.config, "pool_op_kernel_sizes", DEFAULT_POOL_OP_KERNEL_SIZES),
            num_processes_da=getattr(self.config, "num_processes_da", None),
            pin_memory=self.device.type == "cuda",
            intensity_channels=getattr(self.config, "intensity_channels", [0]),
            load_weak_confidence=(self.enable_confidence_ignore or self.confidence_weight),
            left_right_pairs_csv=getattr(self.config, "left_right_pairs_csv", None),
        )

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.real_lr_scheduler.step(epoch - self.lr_scheduler_start_epoch)
            if self.need_synth_for_epoch(epoch):
                self.start_synth_prefetcher()
            else:
                self.stop_synth_prefetcher()
            self.real_net.train()
            self.synth_net.train()

            if epoch == 0:
                self.print_to_log_file(f"stage 1 synth training: epoch 0 to {self.start_weak}")
                set_requires_grad(self.synth_net.network.backbone, True)
                set_requires_grad(self.real_net.network.backbone, False)
                set_requires_grad(self.real_net.network.decoder, True)
            if epoch == self.start_weak:
                weak_stage_name = "synth + weak" if self.enable_weak_synth_training else "weak only"
                self.print_to_log_file(f"stage 2 {weak_stage_name} training: epoch {self.start_weak} to {self.start_one}")
                self.real_net.network.backbone.load_state_dict(self.synth_net.network.backbone.state_dict())
                set_requires_grad(self.synth_net.network.backbone, True)
                set_requires_grad(self.real_net.network.backbone, True)
                set_requires_grad(self.real_net.network.decoder, True)
            if epoch == self.start_one:
                self.print_to_log_file(f"stage 3 single-data fine-tune: epoch {self.start_one} to {self.num_epochs}")
                self.reset_optimizer_and_scheduler(self.trainone_lr, self.num_epochs - self.start_one, self.start_one)
                self.init_trainone_ema()
                self.num_iterations_per_epoch = getattr(self.config, "num_iterations_finetune", 150)
                set_requires_grad(self.synth_net.network.backbone, False)
                set_requires_grad(self.real_net.network.backbone, self.trainone_finetune_scope == "all")
                set_requires_grad(self.real_net.network.decoder, True)
                self.real_net.train()

            if epoch >= self.start_one and self.trainone_finetune_scope == "decoder":
                self.real_net.network.backbone.eval()

            self.assert_shared_decoder()
            outputs = []
            epoch_start = time()
            for _ in range(self.num_iterations_per_epoch):
                try:
                    if epoch < self.start_weak:
                        outputs.append(self.train_synth_batch())
                    elif epoch < self.start_one:
                        outputs.append(self.train_weak_batch())
                    else:
                        outputs.append(self.train_one_batch())
                except RuntimeError as e:
                    msg = str(e)
                    if "background workers" not in msg:
                        raise
                    self.print_to_log_file(
                        "[dataloader] background workers failed; falling back to single-process loader. Error:", msg
                    )
                    self.weakloader, self.simulateloader = get_data_loader(
                        self.config.weakdir,
                        self.config.simulatedir,
                        self.dataset_json_path,
                        tuple(self.config.patch_size),
                        self.config.batch_size,
                        oversample_foreground_percent=self.oversample_foreground_percent,
                        deep_supervision=getattr(self.config, "deep_supervision", False),
                        pool_op_kernel_sizes=getattr(self.config, "pool_op_kernel_sizes", DEFAULT_POOL_OP_KERNEL_SIZES),
                        num_processes_da=0,
                        pin_memory=False,
                        intensity_channels=getattr(self.config, "intensity_channels", [0]),
                        load_weak_confidence=(self.enable_confidence_ignore or self.confidence_weight),
                        left_right_pairs_csv=getattr(self.config, "left_right_pairs_csv", None),
                    )
                    if epoch < self.start_weak:
                        outputs.append(self.train_synth_batch())
                    elif epoch < self.start_one:
                        outputs.append(self.train_weak_batch())
                    else:
                        outputs.append(self.train_one_batch())
            logs = collate_outputs(outputs)
            loss = float(np.mean(logs["loss"]))
            weakloss = float(np.mean(logs["weakloss"]))
            fdloss = float(np.mean(logs.get("fdloss", np.array([0], dtype=np.float32))))
            finetune_loss = float(np.mean(logs["loss_finetune"]))
            self.logger.log("train_losses_synth", loss, epoch)
            self.logger.log("train_losses_weak", weakloss, epoch)
            self.logger.log("train_losses_fd", fdloss, epoch)
            self.logger.log("train_losses_finetune", finetune_loss, epoch)
            self.print_to_log_file(
                f"Epoch {epoch}: synth={loss:.4f}, weak={weakloss:.4f}, fd={fdloss:.4f}, finetune={finetune_loss:.4f}, "
                f"time={time() - epoch_start:.2f}s"
            )
            self.logger.plot_progress_png(self.output_folder)

            if (epoch + 1) % self.save_every == 0:
                if epoch <= self.start_one:
                    self.save_synth_checkpoint(os.path.join(self.output_folder, f"checkpoint_synth_epoch_{epoch}.pth"))
                if epoch >= self.start_weak:
                    self.save_real_checkpoint(os.path.join(self.output_folder, f"checkpoint_real_epoch_{epoch}.pth"))
                if epoch >= self.start_one:
                    self.save_trainone_ema_checkpoint(
                        os.path.join(self.output_folder, f"checkpoint_real_epoch_{epoch}_only_finetune_ema.pth")
                    )
        self.stop_synth_prefetcher()
