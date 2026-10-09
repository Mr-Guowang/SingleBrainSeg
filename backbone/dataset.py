from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import blosc2
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class CaseFiles:
    identifier: str
    image: str
    seg: str
    confidence: Optional[str] = None


def _as_channel_first(array: np.ndarray, dtype) -> np.ndarray:
    array = np.asarray(array, dtype=dtype)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"Expected 3D or channel-first 4D array, got shape {array.shape}")
    return np.ascontiguousarray(array)


def load_nii(path: str, dtype) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(), dtype=dtype)


def zscore_2_98(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    valid = np.isfinite(out) & (out != 0)
    vals = out[valid]
    if vals.size == 0:
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    p2, p98 = np.percentile(vals, [2, 98])
    mask = valid & (out >= p2) & (out <= p98)
    if not np.any(mask):
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = (out - out[mask].mean()) / (out[mask].std() + eps)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def zscore_nnunet(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.nan_to_num(data.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    mean = out.mean()
    std = out.std()
    out -= mean
    out /= max(std, eps)
    return out.astype(np.float32, copy=False)


def compute_class_locations(
    seg: np.ndarray,
    labels: Sequence[int],
    ignore_label: Optional[int] = None,
    min_num_samples: int = 10000,
    min_percent_coverage: float = 0.01,
    seed: int = 1234,
) -> Dict:
    """Sample foreground locations like nnU-Net instead of storing all voxel coordinates."""
    seg0 = seg[0] if seg.ndim == 4 else seg
    classes_or_regions = [int(i) for i in labels if int(i) != 0]
    if ignore_label is not None:
        classes_or_regions.append(tuple([-1] + [int(i) for i in labels]))

    rndst = np.random.RandomState(seed)
    locations = {}

    normalized = []
    requested_labels = set()
    for c in classes_or_regions:
        if isinstance(c, (tuple, list)):
            labs = tuple(int(x) for x in c if int(x) != -1)
            normalized.append(tuple(int(x) for x in c))
            requested_labels.update(labs)
        else:
            lab = int(c)
            normalized.append(lab)
            requested_labels.add(lab)

    if len(requested_labels) == 0:
        return {c: [] for c in normalized}

    valid_mask = np.isin(seg0, np.fromiter(requested_labels, dtype=np.int32))
    coords = np.argwhere(valid_mask)
    seg_sel = seg0[valid_mask]
    del valid_mask

    if seg_sel.size == 0:
        return {c: [] for c in normalized}

    order = np.argsort(seg_sel, kind="stable")
    lab_sorted = seg_sel[order]
    coords_sorted = coords[order]

    change = np.flatnonzero(lab_sorted[1:] != lab_sorted[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, lab_sorted.size]
    labels_present = lab_sorted[starts]
    label_to_range = {int(l): (int(s), int(e)) for l, s, e in zip(labels_present, starts, ends)}
    present_labels = set(label_to_range.keys())

    for c in normalized:
        is_region = isinstance(c, tuple)
        labs = tuple(int(x) for x in c if int(x) != -1) if is_region else (int(c),)
        key = c if is_region else labs[0]
        if not any(lab in present_labels for lab in labs):
            locations[key] = []
            continue

        ranges = []
        counts = []
        for lab in labs:
            r = label_to_range.get(lab)
            if r is None:
                continue
            start, end = r
            count = end - start
            if count > 0:
                ranges.append((start, end))
                counts.append(count)

        total = int(np.sum(counts))
        target_num_samples = min(int(min_num_samples), total)
        target_num_samples = max(target_num_samples, int(np.ceil(total * float(min_percent_coverage))))
        offsets = rndst.choice(total, target_num_samples, replace=False)
        cum = np.cumsum(counts)
        which = np.searchsorted(cum, offsets, side="right")
        prev = np.concatenate(([0], cum[:-1]))
        in_range = offsets - prev[which]
        starts_for_pick = np.fromiter((ranges[i][0] for i in which), dtype=np.int64, count=which.size)
        picked_idx = starts_for_pick + in_range.astype(np.int64)
        selected = coords_sorted[picked_idx]
        locations[key] = np.column_stack((np.zeros(len(selected), dtype=np.int16), selected.astype(np.int16))).astype(np.int16)

    return locations


@dataclass(frozen=True)
class PreprocessedCaseFiles:
    identifier: str
    channels: Tuple[str, ...]
    seg: str
    properties: str
    data_b2nd: Optional[str] = None
    seg_b2nd: Optional[str] = None


class PreprocessedPatchDataset:
    """
    Fast nnU-Net-like dataset backed by .npy channels and precomputed properties.

    Expected layout:
      imagesTr/case_0000.npy, imagesTr/case_0001.npy, ...
      labelsTr/case.npy
      properties/case.pkl

    confidence_channel is kept out of model input and returned separately.
    """

    def __init__(
        self,
        cases: Sequence[PreprocessedCaseFiles],
        labels: Sequence[int],
        ignore_label: Optional[int] = None,
        data_channels: Optional[Sequence[int]] = None,
        confidence_channel: Optional[int] = 1,
        mmap_mode: Optional[str] = "r",
    ):
        if len(cases) == 0:
            raise ValueError("PreprocessedPatchDataset got no cases")
        self.cases = {c.identifier: c for c in cases}
        self.identifiers = sorted(self.cases)
        self.labels = [int(i) for i in labels]
        self.ignore_label = ignore_label
        self.confidence_channel = confidence_channel
        self.mmap_mode = mmap_mode
        first = cases[0]
        if first.data_b2nd is not None:
            n_channels = int(blosc2.open(urlpath=first.data_b2nd, mode="r", dparams={"nthreads": 1}).shape[0])
        else:
            n_channels = len(first.channels)
        if data_channels is None:
            self.data_channels = [i for i in range(n_channels) if i != confidence_channel]
        else:
            self.data_channels = [int(i) for i in data_channels]

    def load_case(self, identifier: str):
        case = self.cases[identifier]
        if case.data_b2nd is not None:
            dparams = {"nthreads": 1}
            mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
            data_all = blosc2.open(urlpath=case.data_b2nd, mode="r", dparams=dparams, **mmap_kwargs)
            if self.data_channels == list(range(data_all.shape[0])):
                data = data_all
            elif len(self.data_channels) == 1:
                c = self.data_channels[0]
                data = data_all[c : c + 1]
            else:
                data = np.asarray(data_all)[self.data_channels]
            confidence = None
            if self.confidence_channel is not None:
                c = int(self.confidence_channel)
                if c >= data_all.shape[0]:
                    raise ValueError(f"confidence_channel={c} but {case.data_b2nd} only has {data_all.shape[0]} channels")
                confidence = data_all[c : c + 1]
            seg = blosc2.open(urlpath=case.seg_b2nd, mode="r", dparams=dparams, **mmap_kwargs)
        else:
            loaded_channels = [np.load(path, mmap_mode=self.mmap_mode) for path in case.channels]
            if len(self.data_channels) == 1:
                data = loaded_channels[self.data_channels[0]][None]
            else:
                data = np.stack([loaded_channels[i] for i in self.data_channels])
            data = data.astype(np.float32, copy=False)
            confidence = None
            if self.confidence_channel is not None and self.confidence_channel < len(loaded_channels):
                confidence = loaded_channels[self.confidence_channel][None].astype(np.float32, copy=False)
            seg = np.load(case.seg, mmap_mode=self.mmap_mode)[None].astype(np.int16, copy=False)
        with open(case.properties, "rb") as f:
            properties = pickle.load(f)
        properties = dict(properties)
        properties.setdefault("identifier", identifier)
        properties.setdefault("shape", tuple(int(i) for i in data.shape[1:]))
        return data, seg, confidence, properties


class NiftiPatchDataset:
    """
    Minimal nnU-Net-style dataset.

    load_case returns (data, seg, confidence, properties), where data is CXYZ,
    seg is 1XYZ, and properties contains nnU-Net class_locations.
    """

    def __init__(
        self,
        cases: Sequence[CaseFiles],
        labels: Sequence[int],
        ignore_label: Optional[int] = None,
        normalize: bool = False,
    ):
        if len(cases) == 0:
            raise ValueError("NiftiPatchDataset got no cases")
        self.cases = {c.identifier: c for c in cases}
        self.identifiers = sorted(self.cases)
        self.labels = [int(i) for i in labels]
        self.ignore_label = ignore_label
        self.normalize = normalize

    def load_case(self, identifier: str):
        case = self.cases[identifier]
        data = _as_channel_first(load_nii(case.image, np.float32), np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        if self.normalize:
            data = np.stack([zscore_nnunet(c) for c in data]).astype(np.float32)
        seg = _as_channel_first(load_nii(case.seg, np.int16), np.int16)
        confidence = None
        if case.confidence is not None:
            confidence = _as_channel_first(load_nii(case.confidence, np.float32), np.float32)
            confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        properties = {
            "class_locations": compute_class_locations(seg, self.labels, self.ignore_label),
            "shape": data.shape[1:],
            "identifier": identifier,
        }
        return data, seg, confidence, properties


def discover_weak_cases(folder: str, prefer_raw_image: bool = False) -> List[CaseFiles]:
    root = Path(folder)
    subjects = [p.name for p in root.iterdir() if p.is_dir()]
    private = [s for s in subjects if "Huashan" in s]
    weighted = subjects + 0 * private

    def image_path(subject: str) -> str:
        nii_dir = root / subject / "nii"
        raw = nii_dir / "image_raw.nii.gz"
        if prefer_raw_image and raw.exists():
            return str(raw)
        return str(nii_dir / "image_zscore.nii.gz")

    return [
        CaseFiles(
            identifier=f"weak_{idx:05d}_{subject}",
            image=image_path(subject),
            seg=str(root / subject / "nii" / "label_raw.nii.gz"),
            confidence=str(root / subject / "nii" / "assigned_label_confidence_qG.nii.gz"),
        )
        for idx, subject in enumerate(weighted)
    ]


def discover_simulate_cases(folder: str) -> List[CaseFiles]:
    root = Path(folder)
    public_root = root / "simulate_public"
    private_root = root / "simulate_private"
    public = [p.name for p in public_root.iterdir() if p.is_dir()] if public_root.exists() else []
    private = [p.name for p in private_root.iterdir() if p.is_dir()] if private_root.exists() else []

    cases: List[CaseFiles] = []
    for idx, subject in enumerate(public):
        cases.append(
            CaseFiles(
                identifier=f"simulate_public_{idx:05d}_{subject}",
                image=str(public_root / subject / f"HCP206_2_{subject}_final.nii.gz"),
                seg=str(public_root / subject / f"HCP206_2_{subject}_source_seg_final.nii.gz"),
            )
        )
    for rep, subject in enumerate(private * 1):
        cases.append(
            CaseFiles(
                identifier=f"simulate_private_{rep:05d}_{subject}",
                image=str(private_root / subject / f"HCP206_2_{subject}_final.nii.gz"),
                seg=str(private_root / subject / f"HCP206_2_{subject}_source_seg_final.nii.gz"),
            )
        )
    return cases


def is_preprocessed_dataset(folder: str) -> bool:
    root = Path(folder)
    has_legacy_npy = (root / "imagesTr").is_dir() and (root / "labelsTr").is_dir() and (root / "properties").is_dir()
    has_b2nd = any(root.glob("*.b2nd")) and any(root.glob("*.pkl"))
    return has_legacy_npy or has_b2nd


def discover_preprocessed_cases(folder: str) -> List[PreprocessedCaseFiles]:
    root = Path(folder)
    cases: List[PreprocessedCaseFiles] = []
    seen = set()

    # New easy_process layout:
    #   subject_img.b2nd
    #   subject_label.b2nd
    #   subject_img.pkl
    for data_file in sorted(root.glob("*_img.b2nd")):
        identifier = data_file.name[:-5]
        subject = identifier[:-4]
        seg_file = root / f"{subject}_label.b2nd"
        prop_file = root / f"{identifier}.pkl"
        if seg_file.exists() and prop_file.exists():
            cases.append(
                PreprocessedCaseFiles(
                    identifier,
                    tuple(),
                    str(seg_file),
                    str(prop_file),
                    data_b2nd=str(data_file),
                    seg_b2nd=str(seg_file),
                )
            )
            seen.add(identifier)
    if cases:
        return cases

    # Legacy backbone layout:
    #   identifier.b2nd
    #   identifier_seg.b2nd
    #   identifier.pkl
    b2nd_ids = sorted(
        p.name[:-5]
        for p in root.glob("*.b2nd")
        if not p.name.endswith("_seg.b2nd") and not p.name.endswith("_label.b2nd")
    )
    for identifier in b2nd_ids:
        if identifier in seen:
            continue
        data_file = root / f"{identifier}.b2nd"
        seg_file = root / f"{identifier}_seg.b2nd"
        prop_file = root / f"{identifier}.pkl"
        if data_file.exists() and seg_file.exists() and prop_file.exists():
            cases.append(
                PreprocessedCaseFiles(
                    identifier,
                    tuple(),
                    str(seg_file),
                    str(prop_file),
                    data_b2nd=str(data_file),
                    seg_b2nd=str(seg_file),
                )
            )
    if cases:
        return cases

    images = root / "imagesTr"
    labels = root / "labelsTr"
    properties = root / "properties"
    identifiers = sorted({p.name[:-9] for p in images.glob("*_0000.npy")})
    for identifier in identifiers:
        channel_files = []
        idx = 0
        while True:
            channel_file = images / f"{identifier}_{idx:04d}.npy"
            if not channel_file.exists():
                break
            channel_files.append(str(channel_file))
            idx += 1
        seg_file = labels / f"{identifier}.npy"
        prop_file = properties / f"{identifier}.pkl"
        if channel_files and seg_file.exists() and prop_file.exists():
            cases.append(PreprocessedCaseFiles(identifier, tuple(channel_files), str(seg_file), str(prop_file)))
    if len(cases) == 0:
        raise ValueError(f"No preprocessed cases found in {folder}")
    return cases


def load_preprocessed_dataset_metadata(folder: str) -> dict:
    metadata_file = Path(folder) / "preprocessed_dataset.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}
