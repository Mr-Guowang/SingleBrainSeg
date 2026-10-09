from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.backends import cudnn

from backbone.trainer_paper import PaperTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results"
DEFAULT_DATASET_JSON = PROJECT_ROOT / "lookuptable" / "dataset.json"
DEFAULT_CONFIDENCE_ITER_SCRIPT = PROJECT_ROOT / "online_iteration" / "iter_online.py"


def run_training(dataset, output_path, dataset_json, device=torch.device("cuda"), config=None):
    trainer = PaperTrainer(dataset, output_path, dataset_json, device, config)
    if torch.cuda.is_available():
        cudnn.deterministic = False
        cudnn.benchmark = True
    trainer.run_training()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paper training pipeline: 100 epochs supervised warmup, then 300 epochs "
            "pseudo-label learning with synthetic teacher regularization and Bayesian "
            "refresh every 10 epochs."
        )
    )

    # Data and output
    parser.add_argument("--dataset", type=str, default="synth2real")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--modelverson", type=str, default="paper")
    parser.add_argument("--taskcode", type=str, default="paper_training")
    parser.add_argument("--dataset_json", type=str, default=str(DEFAULT_DATASET_JSON))
    parser.add_argument("--simulatedir", type=str, required=True, help="Fully supervised simulation/augmentation data for warmup and joint training.")
    parser.add_argument("--weakdir", type=str, required=True, help="Initial pseudo-label training data directory.")
    parser.add_argument("--synthpath", type=str, default="", help="Initial synthetic label source. After each Bayesian refresh this is replaced by output_root/iter_synth.")

    # Model
    parser.add_argument("--modelname", type=str, default="Triad_UNet")
    parser.add_argument("--synth_pretrained", type=str, required=True, help="Initial pretrained checkpoint.")
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=36)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--intensity_channels", type=int, nargs="*", default=[0])
    parser.add_argument("--deep_supervision", action="store_true")
    parser.add_argument("--left_right_pairs_csv", type=str, default=None)

    # Fixed paper schedule; exposed only so experiments can reproduce variants.
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--num_iterations_per_epoch", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--num_processes_da", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "mps"], default="cuda")

    # Regularization used in the joint 300-epoch stage.
    parser.add_argument("--confidence_power", type=float, default=1.0)
    parser.add_argument("--feature_distill_weight", type=float, default=0.1)
    parser.add_argument("--feature_distill_eps", type=float, default=1e-8)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--no_synth_prefetch", dest="enable_synth_prefetch", action="store_false")
    parser.set_defaults(enable_synth_prefetch=True)
    parser.add_argument("--synth_prefetch_size", type=int, default=2)

    # Bayesian pseudo-label iteration.
    parser.add_argument("--confidence_iter_interval", type=int, default=10)
    parser.add_argument("--confidence_iter_script", type=str, default=str(DEFAULT_CONFIDENCE_ITER_SCRIPT))
    parser.add_argument("--confidence_iter_input_csv", type=str, required=True)
    parser.add_argument("--confidence_iter_output_root", type=str, required=True)
    parser.add_argument("--confidence_iter_tissue_csv", type=str, required=True)
    parser.add_argument("--confidence_iter_prior_dir", type=str, required=True)
    parser.add_argument("--confidence_iter_target_spacing", type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument("--confidence_iter_gpu", type=str, default=None)
    parser.add_argument("--no_confidence_iter_force", dest="confidence_iter_force", action="store_false")
    parser.set_defaults(confidence_iter_force=True)

    args = parser.parse_args()

    # Paper defaults: always enabled; not exposed as user-facing switches.
    args.share_encoder = True
    args.enable_ema = True
    args.confidence_weight = True
    args.enable_feature_distill = True
    args.enable_confidence_ignore = False
    args.confidence_ignore_label = 1200
    args.batch_dice = False
    args.fold_right_to_left = False
    args.enable_weak_synth_training = True
    args.weak_synth_interval = 1
    args.enable_affinity = False
    args.iter = 0
    return args


def main():
    args = parse_args()

    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda")
    else:
        device = torch.device("mps")

    output_path = os.path.join(args.output, args.modelverson, f"{args.taskcode}_{args.modelname}")
    os.makedirs(output_path, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict["output_path"] = output_path
    with open(os.path.join(output_path, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)

    run_training(args.dataset, output_path, args.dataset_json, device, args)


if __name__ == "__main__":
    main()
