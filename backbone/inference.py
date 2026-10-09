from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backbone.paths import add_project_paths
else:
    from .paths import add_project_paths

add_project_paths()

from acvl_utils.cropping_and_padding.padding import pad_nd_image  # noqa: E402
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window  # noqa: E402
from backbone.transforms import DEFAULT_LEFT_RIGHT_LABEL_PAIRS, infer_left_right_pairs_from_csv  # noqa: E402


LEFT_RIGHT_LABEL_PAIRS = tuple(DEFAULT_LEFT_RIGHT_LABEL_PAIRS)


def get_left_right_label_pairs_from_config(cfg: dict):
    pairs = infer_left_right_pairs_from_csv(cfg.get("left_right_pairs_csv", None))
    if pairs is None:
        pairs = DEFAULT_LEFT_RIGHT_LABEL_PAIRS
    return tuple((int(left), int(right)) for left, right in pairs)


def swap_left_right_logits(logits: torch.Tensor, pairs=LEFT_RIGHT_LABEL_PAIRS) -> torch.Tensor:
    """Swap left/right class channels in channel-first logits [C, X, Y, Z]."""
    out = logits.clone()
    for left, right in pairs:
        if left < out.shape[0] and right < out.shape[0]:
            out[left] = logits[right]
            out[right] = logits[left]
    return out


@torch.inference_mode()
def _sliding_window_predict_once(
    network: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    device: torch.device,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    autocast_dtype=torch.float16,
):
    image = image.to(device, non_blocking=True)
    data, slicer_revert_padding = pad_nd_image(image, patch_size, "constant", {"value": 0}, True, None)
    spatial_shape = tuple(data.shape[1:])
    steps = compute_steps_for_sliding_window(spatial_shape, tuple(patch_size), tile_step_size)
    slicers = [tuple([slice(None), *[slice(i, i + p) for i, p in zip(step, patch_size)]]) for step in product(*steps)]

    predicted_logits = None
    n_predictions = None
    gaussian = compute_gaussian(tuple(patch_size), dtype=autocast_dtype, device=device) if use_gaussian else None

    for sl in slicers:
        patch = data[sl][None]
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=autocast_dtype):
            logits = network(patch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
        logits = logits[0].float()
        if predicted_logits is None:
            predicted_logits = torch.zeros((logits.shape[0], *spatial_shape), dtype=torch.float32, device=device)
            n_predictions = torch.zeros(spatial_shape, dtype=torch.float32, device=device)
        weight = gaussian if gaussian is not None else 1
        predicted_logits[sl] += logits * weight
        n_predictions[sl[1:]] += weight

    predicted_logits /= n_predictions.clamp_min(1e-8)
    return predicted_logits[(slice(None), *slicer_revert_padding[1:])]


@torch.inference_mode()
def sliding_window_predict(
    network: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Sequence[int],
    device: torch.device,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    mirror_axes=(0,),
    autocast_dtype=torch.float16,
    left_right_label_pairs=LEFT_RIGHT_LABEL_PAIRS,
):
    """nnU-Net-style tiled prediction for a single image tensor [C, X, Y, Z].

    Test-time augmentation is intentionally restricted to left-right mirroring only:
    original prediction + LR-flipped prediction, then flip back spatially, swap
    left/right class channels, and average logits. No AP/SI mirroring is used.
    """
    if mirror_axes is None:
        mirror_axes = ()
    mirror_axes = tuple(mirror_axes)
    unsupported_axes = tuple(ax for ax in mirror_axes if ax != 0)
    if unsupported_axes:
        raise ValueError(f"Only left-right mirror axis 0 is supported, got mirror_axes={mirror_axes}")

    was_training = network.training
    network.eval()
    logits = _sliding_window_predict_once(
        network, image, patch_size, device, tile_step_size, use_gaussian, autocast_dtype
    )
    if 0 in mirror_axes:
        flipped_image = torch.flip(image, dims=(1,))
        flipped_logits = _sliding_window_predict_once(
            network, flipped_image, patch_size, device, tile_step_size, use_gaussian, autocast_dtype
        )
        flipped_logits = torch.flip(flipped_logits, dims=(1,))
        flipped_logits = swap_left_right_logits(flipped_logits, left_right_label_pairs)
        logits = 0.5 * (logits + flipped_logits)
    if was_training:
        network.train()
    return logits


def parse_args():
    parser = argparse.ArgumentParser(description="Single-case GG-BondNet inference.")
    parser.add_argument("--args_json", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Input 3D/4D NIfTI image.")
    parser.add_argument("--out", type=str, default=None, help="Output segmentation NIfTI path.")
    parser.add_argument("--output_folder", type=str, default=None, help="Alternative to --out; saves {image_stem}_pred.nii.gz here.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, or filename relative to args_json output_path.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tile_step_size", type=float, default=0.5)
    parser.add_argument("--no_tta", action="store_true", help="Disable left-right flip TTA.")
    parser.add_argument("--normalize", choices=["nnunet", "old_percentile", "none"], default="nnunet")
    parser.add_argument("--post", action="store_true", help="Apply connected-component post-processing before saving.")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None, help="Resample image to this spacing before inference, then restore prediction to original image shape.")
    return parser.parse_args()


def nii_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def resolve_output_path(args) -> Path:
    if args.out is None and args.output_folder is None:
        raise ValueError("Please provide either --out or --output_folder")
    if args.out is not None:
        return Path(args.out)
    out_dir = Path(args.output_folder)
    return out_dir / f"{nii_stem(Path(args.image))}_pred.nii.gz"


def load_config(args_json: str) -> dict:
    with open(args_json, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("network_weights", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def set_deep_supervision_enabled(model, enabled: bool):
    if hasattr(model, "deep_supervision"):
        model.deep_supervision = enabled
    net = getattr(model, "network", None)
    if net is not None and hasattr(net, "deep_supervision"):
        net.deep_supervision = enabled
    decoder = getattr(net, "decoder", None) if net is not None else None
    if decoder is not None and hasattr(decoder, "deep_supervision"):
        decoder.deep_supervision = enabled
    if decoder is not None and hasattr(decoder, "do_ds"):
        decoder.do_ds = enabled


def strip_module_prefix(state_dict: dict, model_state: dict) -> dict:
    fixed = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key not in model_state and new_key.startswith("module."):
            new_key = new_key[7:]
        fixed[new_key] = value
    return fixed


def resolve_checkpoint_path(checkpoint: str, cfg: dict) -> str:
    ckpt = Path(checkpoint)
    if ckpt.is_absolute():
        return str(ckpt)
    output_path = cfg.get("output_path", None)
    if output_path is not None:
        candidate = Path(output_path) / checkpoint
        if candidate.exists():
            return str(candidate)
    return str(ckpt)


def normalize_image(data: np.ndarray, mode: str) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if mode == "none":
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mode == "old_percentile":
        p2, p98 = np.percentile(data, (2, 98))
        data = np.clip(data, p2, p98)
        mean = float(data.mean())
        std = float(data.std())
        data = data - mean if std < 1e-8 else (data - mean) / std
        return np.clip(data, -3, 3).astype(np.float32)
    mean = float(data.mean())
    std = float(data.std())
    data = data - mean if std < 1e-8 else (data - mean) / std
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def to_channel_first(data: np.ndarray, expected_channels: int) -> np.ndarray:
    if data.ndim == 3:
        data = data[None]
    elif data.ndim == 4:
        if data.shape[0] == expected_channels:
            pass
        elif data.shape[-1] == expected_channels:
            data = np.moveaxis(data, -1, 0)
        elif data.shape[0] <= 16:
            pass
        elif data.shape[-1] <= 16:
            data = np.moveaxis(data, -1, 0)
        else:
            raise ValueError(f"Cannot infer channel axis for shape {data.shape}")
    else:
        raise ValueError(f"Expected 3D or 4D image, got shape {data.shape}")
    if data.shape[0] != expected_channels:
        raise ValueError(f"Expected {expected_channels} input channel(s), got {data.shape[0]} for shape {data.shape}")
    return np.ascontiguousarray(data)


def compute_new_shape(old_shape, old_spacing, new_spacing):
    return tuple(max(1, int(round(float(osz) * float(osp) / float(nsp)))) for osz, osp, nsp in zip(old_shape, old_spacing, new_spacing))


def resize_to_shape(array: np.ndarray, target_shape, order: int) -> np.ndarray:
    target_shape = tuple(int(i) for i in target_shape)
    if tuple(array.shape) == target_shape:
        return np.ascontiguousarray(array)
    factors = [t / s for t, s in zip(target_shape, array.shape)]
    out = ndi_zoom(array, factors, order=order)
    if tuple(out.shape) == target_shape:
        return np.ascontiguousarray(out)
    fixed = np.zeros(target_shape, dtype=out.dtype)
    common = tuple(slice(0, min(out.shape[i], target_shape[i])) for i in range(3))
    fixed[common] = out[common]
    return np.ascontiguousarray(fixed)


def resample_channel_first(data: np.ndarray, current_spacing, target_spacing, order: int = 3) -> np.ndarray:
    if target_spacing is None:
        return np.ascontiguousarray(data)
    target_shape = compute_new_shape(data.shape[1:], current_spacing, target_spacing)
    if tuple(data.shape[1:]) == target_shape:
        return np.ascontiguousarray(data)
    return np.stack([resize_to_shape(channel, target_shape, order=order) for channel in data], axis=0).astype(np.float32, copy=False)


def restore_prediction_to_original_shape(pred: np.ndarray, original_shape) -> np.ndarray:
    pred = np.asarray(pred)
    if tuple(pred.shape) == tuple(original_shape):
        return np.ascontiguousarray(pred)
    return resize_to_shape(pred.astype(np.float32, copy=False), original_shape, order=0).astype(pred.dtype, copy=False)


def main():
    args = parse_args()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from backbone.getmodel import build_model
    from backbone.paths import add_project_paths
    from backbone.post_process import post_process_segmentation_array

    add_project_paths()
    cfg = load_config(args.args_json)

    modelname = cfg.get("modelname", "Triad_UNet")
    in_channels = int(cfg.get("in_channels", 1))
    num_classes = int(cfg.get("num_classes", 36))
    patch_size = tuple(int(i) for i in cfg.get("patch_size", [128, 128, 128]))
    left_right_label_pairs = get_left_right_label_pairs_from_config(cfg)

    device = torch.device("cuda", 0) if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = True

    network = build_model(modelname=modelname, in_channels=in_channels, num_classes=num_classes)
    set_deep_supervision_enabled(network, False)

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(unwrap_checkpoint_state(checkpoint), network.state_dict())
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    network.to(device)
    network.eval()

    image_path = Path(args.image)
    out_path = resolve_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[model] {modelname}, in_channels={in_channels}, num_classes={num_classes}, patch_size={patch_size}")
    print(f"[tta] left_right_label_pairs={left_right_label_pairs}")
    print(f"[checkpoint] loaded {checkpoint_path}; missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[checkpoint] missing keys sample: {missing[:5]}")
    if unexpected:
        print(f"[checkpoint] unexpected keys sample: {unexpected[:5]}")

    target_spacing = None if args.target_spacing is None else tuple(float(i) for i in args.target_spacing)
    if target_spacing is not None:
        print(f"[resample] input -> target_spacing={target_spacing}; prediction -> original image shape")
    if args.post:
        print("[post] enabled, max_radius=3")

    img = nib.load(str(image_path))
    original_spacing = tuple(float(i) for i in img.header.get_zooms()[:3])
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    data = to_channel_first(data, in_channels)
    original_shape = tuple(int(i) for i in data.shape[1:])
    data = resample_channel_first(data, original_spacing, target_spacing, order=3)
    data = np.stack([normalize_image(channel, args.normalize) for channel in data], axis=0).astype(np.float32, copy=False)

    image = torch.from_numpy(data)
    mirror_axes = () if args.no_tta else (0,)
    logits = sliding_window_predict(
        network=network,
        image=image,
        patch_size=patch_size,
        device=device,
        tile_step_size=args.tile_step_size,
        use_gaussian=True,
        mirror_axes=mirror_axes,
        left_right_label_pairs=left_right_label_pairs,
    )
    pred = torch.argmax(logits, dim=0)
    if args.post:
        pred = post_process_segmentation_array(pred, max_radius=3)
    else:
        pred = pred.cpu().numpy()
    pred = restore_prediction_to_original_shape(pred, original_shape)
    pred = pred.astype(np.uint16, copy=False)

    header = img.header.copy()
    header.set_data_dtype(np.uint16)
    nib.save(nib.Nifti1Image(pred, img.affine, header), str(out_path))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
