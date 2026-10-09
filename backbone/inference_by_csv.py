from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom as ndi_zoom

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_json", type=str, required=True)
    parser.add_argument("--input_csv", type=str, default = '/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/table/brainseg_test_final.csv')
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default='checkpoint_real_epoch_1199_only_finetune_ema.pth')
    parser.add_argument("--gpu", type=str, default="6")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tile_step_size", type=float, default=0.5)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--normalize", choices=["nnunet", "old_percentile", "none"], default="nnunet")
    parser.add_argument("--post", action="store_true", help="Apply connected-component post-processing before saving.")
    parser.add_argument("--lookuptable_csv", type=str, default=None, help="Brain_lookuptable.csv for post-processing. Defaults to args_json left_right_pairs_csv.")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None, help="Resample each input image to this spacing before inference, then resample prediction back to original shape.")
    return parser.parse_args()


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


def iter_nii_files(folder: str):
    root = Path(folder)
    files = sorted([p for p in root.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))])
    for file in files:
        yield file


def nii_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    if path.name.endswith(".nii"):
        return path.name[:-4]
    return path.stem


def main():
    args = parse_args()
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    if __package__ is None or __package__ == "":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from backbone.getmodel import build_model
    from backbone.inference import get_left_right_label_pairs_from_config, sliding_window_predict
    from backbone.paths import add_project_paths
    from backbone.post_process import post_process_segmentation_array, resolve_lookuptable_csv

    add_project_paths()

    cfg = load_config(args.args_json)

    modelname = cfg.get("modelname", "Triad_UNet")
    in_channels = int(cfg.get("in_channels", 1))
    num_classes = int(cfg.get("num_classes", 36))
    patch_size = tuple(int(i) for i in cfg.get("patch_size", [128, 128, 128]))
    left_right_label_pairs = get_left_right_label_pairs_from_config(cfg)
    post_lookuptable_csv = resolve_lookuptable_csv(args.lookuptable_csv, args.args_json)

    device = torch.device("cuda", 0) if args.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark = True

    network = build_model(modelname=modelname, in_channels=in_channels, num_classes=num_classes)
    set_deep_supervision_enabled(network, False)
    
    checkpoint = torch.load(os.path.join(cfg.get("output_path"),args.checkpoint), map_location=device, weights_only=False)
    state_dict = strip_module_prefix(unwrap_checkpoint_state(checkpoint), network.state_dict())
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    network.to(device)
    network.eval()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"[model] {modelname}, in_channels={in_channels}, num_classes={num_classes}, patch_size={patch_size}")
    print(f"[tta] left_right_label_pairs={left_right_label_pairs}")
    print(f"[checkpoint] loaded {args.checkpoint}; missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[checkpoint] missing keys sample: {missing[:5]}")
    if unexpected:
        print(f"[checkpoint] unexpected keys sample: {unexpected[:5]}")
    if args.post:
        print(f"[post] enabled, max_radius=3, lookuptable={post_lookuptable_csv}")
    target_spacing = None if args.target_spacing is None else tuple(float(i) for i in args.target_spacing)
    if target_spacing is not None:
        print(f"[resample] input -> target_spacing={target_spacing}; prediction -> original image shape")

    mirror_axes = () if args.no_tta else (0,)
    import pandas as pd
    df = pd.read_csv(args.input_csv)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        Site,SubjectID,Session = row['Site'],row['SubjectID'],row['Session']
        group = row['Group']
        Session = str(Session)
        SubjectID = str(SubjectID)
        Site = str(Site)
        data_path = row['process']

        image_path = os.path.join(data_path,'step_1_T1w_process_ANTs-2.4.0_synthmorph/T1w2MNI_RigidWarped.nii.gz')

        if not os.path.exists(image_path):
            image_path = os.path.join(data_path,'step_1_T1w_process_ANTs-2.4.0/T1w2MNI_RigidWarped.nii.gz')
        
        img = nib.load(str(image_path))
        original_spacing = tuple(float(i) for i in img.header.get_zooms()[:3])
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        data = to_channel_first(data, in_channels)
        original_shape = tuple(int(i) for i in data.shape[1:])
        data = resample_channel_first(data, original_spacing, target_spacing, order=3)
        data = np.stack([normalize_image(channel, args.normalize) for channel in data], axis=0).astype(np.float32, copy=False)
        image = torch.from_numpy(data)
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
            pred = post_process_segmentation_array(pred, lookuptable_csv=post_lookuptable_csv, max_radius=3)
        else:
            pred = pred.cpu().numpy()
        pred = restore_prediction_to_original_shape(pred, original_shape)
        pred = pred.astype(np.uint16, copy=False)
        if args.post:
            out_path = os.path.join(args.output_folder,Site,SubjectID,Session,f'{Site}_{SubjectID}_{Session}_ggbond_post.nii.gz')
        else:
            out_path = os.path.join(args.output_folder,Site,SubjectID,Session,f'{Site}_{SubjectID}_{Session}_ggbond.nii.gz')
        os.makedirs(os.path.join(args.output_folder,Site,SubjectID,Session),exist_ok=True)
        nib.save(nib.Nifti1Image(pred, img.affine, img.header), str(out_path))
        print(f"[{index}/{len(df)}] saved {out_path}")


if __name__ == "__main__":
    main()
