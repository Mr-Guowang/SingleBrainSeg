from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import blosc2
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as ndi_zoom

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backbone.dataset import compute_class_locations, zscore_nnunet  # noqa: E402
from backbone.get_data_loader import load_dataset_json  # noqa: E402
from backbone.label_manager import build_label_manager  # noqa: E402


DEFAULT_DATASET_JSON = str(Path(__file__).resolve().parents[1] / "lookuptable" / "dataset.json")
IMG_RE = re.compile(r"(.+)_img_(\d{4})\.nii(?:\.gz)?$")
LABEL_RE = re.compile(r"(.+)_label\.nii(?:\.gz)?$")


@dataclass(frozen=True)
class CaseSpec:
    subject: str
    images: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class ProcessConfig:
    output_folder: str
    dataset_json: str
    target_spacing: tuple[float, float, float] | None
    no_zscore_channels: tuple[int, ...]
    data_channels: tuple[int, ...] | None
    confidence_channel: int | None
    image_interp_order: int
    aux_interp_order: int
    overwrite: bool


def discover_cases(input_folder: str) -> list[CaseSpec]:
    root = Path(input_folder)
    images: dict[str, dict[int, str]] = {}
    labels: dict[str, str] = {}
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        m = IMG_RE.match(f.name)
        if m is not None:
            subject, idx = m.group(1), int(m.group(2))
            images.setdefault(subject, {})[idx] = str(f)
            continue
        m = LABEL_RE.match(f.name)
        if m is not None:
            labels[m.group(1)] = str(f)

    cases = []
    for subject in sorted(images):
        if subject not in labels:
            print(f"[skip] {subject}: missing {subject}_label.nii.gz", flush=True)
            continue
        ch_map = images[subject]
        expected = list(range(max(ch_map) + 1))
        missing = [i for i in expected if i not in ch_map]
        if missing:
            raise RuntimeError(f"{subject}: missing image channels {missing}")
        cases.append(CaseSpec(subject=subject, images=tuple(ch_map[i] for i in expected), label=labels[subject]))
    return cases


def load_nii_3d(path: str, dtype=np.float32):
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=dtype)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image at {path}, got shape {data.shape}")
    spacing = tuple(float(i) for i in img.header.get_zooms()[:3])
    return np.ascontiguousarray(data), spacing


def target_shape_from_spacing(shape: Sequence[int], spacing: Sequence[float], target_spacing: Sequence[float]) -> tuple[int, int, int]:
    return tuple(max(1, int(round(int(s) * float(sp) / float(tsp)))) for s, sp, tsp in zip(shape, spacing, target_spacing))


def resample_to_shape(data: np.ndarray, target_shape: Sequence[int], order: int) -> np.ndarray:
    target_shape = tuple(int(i) for i in target_shape)
    if tuple(data.shape) == target_shape:
        return np.ascontiguousarray(data)
    factors = [t / s for t, s in zip(target_shape, data.shape)]
    out = ndi_zoom(data, factors, order=order)
    if tuple(out.shape) == target_shape:
        return np.ascontiguousarray(out)
    fixed = np.zeros(target_shape, dtype=out.dtype)
    common = tuple(slice(0, min(out.shape[i], target_shape[i])) for i in range(3))
    fixed[common] = out[common]
    return np.ascontiguousarray(fixed)


def maybe_resample(data: np.ndarray, spacing: Sequence[float], target_spacing: Sequence[float] | None, order: int):
    if target_spacing is None:
        return np.ascontiguousarray(data), tuple(float(i) for i in spacing)
    target_shape = target_shape_from_spacing(data.shape, spacing, target_spacing)
    return resample_to_shape(data, target_shape, order), tuple(float(i) for i in target_spacing)


def save_b2nd(array: np.ndarray, path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not overwrite:
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    blosc2.set_nthreads(1)
    blosc2.asarray(
        np.ascontiguousarray(array),
        urlpath=str(path),
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8},
    )


def load_labels(dataset_json: str):
    dataset_info = load_dataset_json(dataset_json)
    label_manager = build_label_manager(dataset_info)
    labels = [int(i) for i in label_manager.all_labels]
    return dataset_info, labels, label_manager.ignore_label


def infer_channels(num_channels: int, no_zscore_channels: Sequence[int], data_channels, confidence_channel):
    no_zscore = {int(i) for i in no_zscore_channels}
    intensity_channels = [i for i in range(num_channels) if i not in no_zscore]
    if confidence_channel is None and len(no_zscore) == 1:
        confidence_channel = sorted(no_zscore)[0]
    if data_channels is None:
        data_channels = [i for i in range(num_channels) if i != confidence_channel]
    return [int(i) for i in data_channels], confidence_channel, intensity_channels


def process_case(case: CaseSpec, cfg: ProcessConfig):
    out = Path(cfg.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    data_id = f"{case.subject}_img"
    label_id = f"{case.subject}_label"
    data_file = out / f"{data_id}.b2nd"
    label_file = out / f"{label_id}.b2nd"
    prop_file = out / f"{data_id}.pkl"

    if not cfg.overwrite and data_file.exists() and label_file.exists() and prop_file.exists():
        return {"subject": case.subject, "skipped": True, "shape": None}

    target_spacing = cfg.target_spacing
    data_channels, confidence_channel, intensity_channels = infer_channels(
        len(case.images), cfg.no_zscore_channels, cfg.data_channels, cfg.confidence_channel
    )

    seg, seg_spacing = load_nii_3d(case.label, np.float32)
    seg = np.rint(seg).astype(np.int16)
    seg, final_spacing = maybe_resample(seg, seg_spacing, target_spacing, order=0)
    seg = np.rint(seg).astype(np.int16)
    final_shape = tuple(int(i) for i in seg.shape)

    channels = []
    source_spacings = []
    source_shapes = []
    for idx, image_path in enumerate(case.images):
        arr, spacing = load_nii_3d(image_path, np.float32)
        source_spacings.append(tuple(float(i) for i in spacing))
        source_shapes.append(tuple(int(i) for i in arr.shape))
        do_zscore = idx in intensity_channels
        order = cfg.image_interp_order if do_zscore else cfg.aux_interp_order
        arr, _ = maybe_resample(arr, spacing, target_spacing, order=order)
        if tuple(arr.shape) != final_shape:
            arr = resample_to_shape(arr, final_shape, order=order)
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if do_zscore:
            arr = zscore_nnunet(arr)
        channels.append(arr.astype(np.float32, copy=False))

    data = np.stack(channels, axis=0).astype(np.float32, copy=False)
    seg_ch = seg[None].astype(np.int16, copy=False)
    dataset_info, labels, ignore_label = load_labels(cfg.dataset_json)
    properties = {
        "class_locations": compute_class_locations(seg_ch, labels, ignore_label),
        "shape": final_shape,
        "identifier": data_id,
        "subject": case.subject,
        "source_modalities": tuple(case.images),
        "source_seg": case.label,
        "source_spacing": source_spacings,
        "source_shape": source_shapes,
        "source_seg_spacing": tuple(float(i) for i in seg_spacing),
        "target_spacing": final_spacing,
        "data_channels": tuple(data_channels),
        "confidence_channel": confidence_channel,
        "intensity_channels": tuple(intensity_channels),
        "normalization_scheme": "nnunet_zscore_per_case_full_volume",
        "normalization_uses_mask": False,
    }

    save_b2nd(data, data_file, cfg.overwrite)
    save_b2nd(seg_ch, label_file, cfg.overwrite)
    with open(prop_file, "wb") as f:
        pickle.dump(properties, f)
    return {
        "subject": case.subject,
        "skipped": False,
        "shape": final_shape,
        "spacing": final_spacing,
        "data_file": str(data_file),
        "label_file": str(label_file),
        "properties": str(prop_file),
        "dataset_info": dataset_info,
        "num_channels": data.shape[0],
        "data_channels": data_channels,
        "confidence_channel": confidence_channel,
        "intensity_channels": intensity_channels,
    }


def write_preprocessed_dataset_json(output_folder: str, dataset_json: str, results: Sequence[dict], target_spacing):
    dataset_info = load_dataset_json(dataset_json)
    valid = [r for r in results if not r.get("skipped")]
    if valid:
        sample = valid[0]
        num_channels = sample["num_channels"]
        data_channels = sample["data_channels"]
        confidence_channel = sample["confidence_channel"]
        intensity_channels = sample["intensity_channels"]
    else:
        # Fall back to existing metadata if all cases were skipped.
        meta_file = Path(output_folder) / "preprocessed_dataset.json"
        if meta_file.exists():
            return
        num_channels = None
        data_channels = None
        confidence_channel = None
        intensity_channels = None

    num_cases = len([p for p in Path(output_folder).glob("*_img.b2nd")])
    metadata = {
        "format": "backbone_easy_folder_b2nd_v1",
        "num_cases": num_cases,
        "num_channels": num_channels,
        "data_channels": data_channels,
        "confidence_channel": confidence_channel,
        "confidence_channels": [] if confidence_channel is None else [int(confidence_channel)],
        "intensity_channels": intensity_channels,
        "all_modalities_in_one_b2nd": True,
        "normalization_scheme": "nnunet_zscore_per_case_full_volume",
        "normalization_uses_mask": False,
        "target_spacing": None if target_spacing is None else [float(i) for i in target_spacing],
        "labels": dataset_info.get("labels", {}),
        "file_ending": ".b2nd",
        "storage": "b2nd",
        "naming": "{subject}_img.b2nd + {subject}_label.b2nd",
    }
    with open(Path(output_folder) / "preprocessed_dataset.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def run_folder(
    input_folder: str,
    output_folder: str,
    dataset_json: str = DEFAULT_DATASET_JSON,
    target_spacing: Sequence[float] | None = None,
    no_zscore_channels: Sequence[int] = (),
    data_channels: Sequence[int] | None = None,
    confidence_channel: int | None = None,
    num_processes: int = 8,
    overwrite: bool = False,
    limit: int | None = None,
):
    cases = discover_cases(input_folder)
    if limit is not None:
        cases = cases[: int(limit)]
    if not cases:
        raise RuntimeError(f"No cases found in {input_folder}. Expected subject_img_0000.nii.gz and subject_label.nii.gz")
    cfg = ProcessConfig(
        output_folder=output_folder,
        dataset_json=dataset_json,
        target_spacing=None if target_spacing is None else tuple(float(i) for i in target_spacing),
        no_zscore_channels=tuple(int(i) for i in no_zscore_channels),
        data_channels=None if data_channels is None else tuple(int(i) for i in data_channels),
        confidence_channel=None if confidence_channel is None else int(confidence_channel),
        image_interp_order=3,
        aux_interp_order=1,
        overwrite=overwrite,
    )
    print(f"Found {len(cases)} cases in {input_folder}")
    print(f"Output -> {output_folder}")
    print(f"target_spacing={cfg.target_spacing}, no_zscore_channels={cfg.no_zscore_channels}, confidence_channel={cfg.confidence_channel}")

    results = []
    if num_processes <= 1:
        for case in cases:
            res = process_case(case, cfg)
            results.append(res)
            print("done", case.subject, "skipped" if res.get("skipped") else res.get("shape"), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(num_processes)) as ex:
            futs = [ex.submit(process_case, case, cfg) for case in cases]
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                print("done", res["subject"], "skipped" if res.get("skipped") else res.get("shape"), flush=True)
    write_preprocessed_dataset_json(output_folder, dataset_json, results, cfg.target_spacing)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None)
    parser.add_argument("--dataset_json", default=DEFAULT_DATASET_JSON)
    parser.add_argument("--no_zscore_channels", type=int, nargs="*", default=[])
    parser.add_argument("--data_channels", type=int, nargs="*", default=None)
    parser.add_argument("--confidence_channel", type=int, default=None)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        dataset_json=args.dataset_json,
        target_spacing=args.target_spacing,
        no_zscore_channels=args.no_zscore_channels,
        data_channels=args.data_channels,
        confidence_channel=args.confidence_channel,
        num_processes=args.num_processes,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
