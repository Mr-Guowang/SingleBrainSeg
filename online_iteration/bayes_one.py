from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_ROOT = PROJECT_ROOT
CONFIDENCE_PY = Path(__file__).resolve().parent / "confidence.py"
DEFAULT_DATASET_JSON = PROJECT_ROOT / "lookuptable" / "dataset.json"
DEFAULT_TISSUE_CSV = Path("/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/table/subspace_table.csv")

if str(BACKBONE_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKBONE_ROOT))

from backbone.dataset import compute_class_locations, zscore_nnunet  # noqa: E402
from backbone.easy_process import (  # noqa: E402
    load_nii_3d,
    maybe_resample,
    resample_to_shape,
    save_b2nd,
    write_preprocessed_dataset_json,
)
from backbone.get_data_loader import load_dataset_json  # noqa: E402
from backbone.label_manager import build_label_manager  # noqa: E402


def load_confidence_module():
    spec = importlib.util.spec_from_file_location("brainseg_subspace_confidence", CONFIDENCE_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import confidence module from {CONFIDENCE_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


confmod = load_confidence_module()


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".b2nd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_subject(subject: str | None, label: str, confidence_dir: str) -> str:
    if subject:
        return subject
    conf_path = Path(confidence_dir)
    if (conf_path / "npy").is_dir():
        return conf_path.name
    label_name = strip_nii_suffix(Path(label))
    for suffix in ("_ggbond_post", "_ggbond", "_label", "_seg"):
        if label_name.endswith(suffix):
            return label_name[: -len(suffix)]
    return label_name


def resolve_confidence_subject_dir(confidence_root_or_subject: str, subject: str) -> Path:
    root = Path(confidence_root_or_subject)
    if (root / "npy" / "posterior_q.npy").exists():
        return root
    subject_dir = root / subject
    if (subject_dir / "npy" / "posterior_q.npy").exists():
        return subject_dir
    raise FileNotFoundError(
        "Could not find posterior_q.npy. Expected either "
        f"{root}/npy/posterior_q.npy or {subject_dir}/npy/posterior_q.npy"
    )


def load_posterior_and_certainty(confidence_dir: Path):
    npy_dir = confidence_dir / "npy"
    posterior_path = npy_dir / "posterior_q.npy"
    certainty_path = npy_dir / "certainty_G.npy"
    uncertainty_path = npy_dir / "uncertainty_entropy_norm.npy"
    if not posterior_path.exists():
        raise FileNotFoundError(f"Missing posterior: {posterior_path}")
    posterior = np.load(posterior_path, mmap_mode="r")
    if posterior.ndim != 4:
        raise ValueError(f"Expected posterior_q.npy to be 4D K,X,Y,Z, got {posterior.shape}")
    if certainty_path.exists():
        certainty = np.load(certainty_path, mmap_mode="r")
    elif uncertainty_path.exists():
        certainty = 1.0 - np.load(uncertainty_path, mmap_mode="r")
    else:
        raise FileNotFoundError(f"Missing certainty_G.npy and uncertainty_entropy_norm.npy in {npy_dir}")
    if tuple(certainty.shape) != tuple(posterior.shape[1:]):
        raise ValueError(f"Certainty/posterior shape mismatch: certainty={certainty.shape}, posterior={posterior.shape}")
    return posterior, certainty


def compute_confidence_from_saved_posterior(label: np.ndarray, confidence_dir: Path, tissue_csv: str) -> np.ndarray:
    tissues, label_groups, _ = confmod.load_tissue_csv(tissue_csv)
    merged_idx, _, _ = confmod.build_merged_label(label, tissues, label_groups)
    posterior, certainty = load_posterior_and_certainty(confidence_dir)
    if posterior.shape[0] != len(tissues):
        raise ValueError(f"Posterior channels={posterior.shape[0]} but tissue csv has {len(tissues)} tissues")
    if tuple(posterior.shape[1:]) != tuple(label.shape):
        raise ValueError(
            f"Posterior/label shape mismatch: posterior={posterior.shape[1:]}, label={label.shape}. "
            "Saved confidence maps must be in the same space as the input label."
        )

    confidence = np.zeros(label.shape, dtype=np.float32)
    certainty_arr = np.asarray(certainty, dtype=np.float32)
    for k in range(len(tissues)):
        mask = merged_idx == k
        if np.any(mask):
            qk = np.asarray(posterior[k], dtype=np.float32)
            confidence[mask] = qk[mask] * certainty_arr[mask]
    return np.clip(np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)


def load_labels(dataset_json: str):
    dataset_info = load_dataset_json(dataset_json)
    label_manager = build_label_manager(dataset_info)
    labels = [int(i) for i in label_manager.all_labels]
    return dataset_info, labels, label_manager.ignore_label


def remove_existing(path: Path):
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def process_one(args) -> dict:
    subject = infer_subject(args.subject, args.label, args.prior)
    confidence_dir = resolve_confidence_subject_dir(args.prior, subject)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    data_file = output_folder / f"{subject}_img.b2nd"
    label_file = output_folder / f"{subject}_label.b2nd"
    prop_file = output_folder / f"{subject}_img.pkl"

    if not args.overwrite and data_file.exists() and label_file.exists() and prop_file.exists():
        return {"subject": subject, "skipped": True, "data_file": str(data_file), "label_file": str(label_file)}

    image, image_spacing = load_nii_3d(args.image, np.float32)
    label, label_spacing = load_nii_3d(args.label, np.float32)
    label = np.rint(label).astype(np.int16)
    if tuple(image.shape) != tuple(label.shape):
        raise ValueError(f"Image/label shape mismatch: image={image.shape}, label={label.shape}")

    confidence_qg = compute_confidence_from_saved_posterior(label, confidence_dir, args.tissue_csv)
    target_spacing = None if args.target_spacing is None else tuple(float(i) for i in args.target_spacing)

    label_rs, final_spacing = maybe_resample(label, label_spacing, target_spacing, order=0)
    label_rs = np.rint(label_rs).astype(np.int16)
    final_shape = tuple(int(i) for i in label_rs.shape)

    image_rs, _ = maybe_resample(image, image_spacing, target_spacing, order=args.image_interp_order)
    if tuple(image_rs.shape) != final_shape:
        image_rs = resample_to_shape(image_rs, final_shape, order=args.image_interp_order)
    image_rs = np.nan_to_num(image_rs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    image_z = zscore_nnunet(image_rs)

    conf_rs, _ = maybe_resample(confidence_qg, image_spacing, target_spacing, order=args.confidence_interp_order)
    if tuple(conf_rs.shape) != final_shape:
        conf_rs = resample_to_shape(conf_rs, final_shape, order=args.confidence_interp_order)
    conf_rs = np.clip(np.nan_to_num(conf_rs.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    data = np.stack((image_z.astype(np.float32, copy=False), conf_rs.astype(np.float32, copy=False)), axis=0)
    seg_ch = label_rs[None].astype(np.int16, copy=False)

    dataset_info, labels, ignore_label = load_labels(args.dataset_json)
    properties = {
        "class_locations": compute_class_locations(seg_ch, labels, ignore_label),
        "shape": final_shape,
        "identifier": f"{subject}_img",
        "subject": subject,
        "source_modalities": (str(args.image), str(confidence_dir)),
        "source_seg": str(args.label),
        "source_spacing": [tuple(float(i) for i in image_spacing), tuple(float(i) for i in image_spacing)],
        "source_shape": [tuple(int(i) for i in image.shape), tuple(int(i) for i in confidence_qg.shape)],
        "source_seg_spacing": tuple(float(i) for i in label_spacing),
        "target_spacing": final_spacing,
        "data_channels": (0,),
        "confidence_channel": 1,
        "intensity_channels": (0,),
        "normalization_scheme": "nnunet_zscore_per_case_full_volume",
        "normalization_uses_mask": False,
        "confidence_source": "saved_posterior_q_times_certainty_G",
    }

    if args.overwrite:
        remove_existing(data_file)
        remove_existing(label_file)
        remove_existing(prop_file)

    save_b2nd(data, data_file, overwrite=True)
    save_b2nd(seg_ch, label_file, overwrite=True)
    with open(prop_file, "wb") as f:
        pickle.dump(properties, f)

    result = {
        "subject": subject,
        "skipped": False,
        "shape": final_shape,
        "spacing": final_spacing,
        "data_file": str(data_file),
        "label_file": str(label_file),
        "properties": str(prop_file),
        "dataset_info": dataset_info,
        "num_channels": 2,
        "data_channels": [0],
        "confidence_channel": 1,
        "intensity_channels": [0],
    }
    write_preprocessed_dataset_json(str(output_folder), args.dataset_json, [result], target_spacing)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Original image in the same space as label and saved confidence maps.")
    parser.add_argument("--label", required=True, help="Current predicted/postprocessed label nii.")
    parser.add_argument("--prior", required=True, help="Saved confidence root or subject dir. Kept as --prior for iter_online compatibility.")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--dataset_json", default=str(DEFAULT_DATASET_JSON))
    parser.add_argument("--tissue_csv", default=str(DEFAULT_TISSUE_CSV))
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[0.7, 0.7, 0.7])
    parser.add_argument("--image_interp_order", type=int, default=3)
    parser.add_argument("--confidence_interp_order", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(process_one(args))


if __name__ == "__main__":
    main()
