from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


ADULT_DEFAULT_LOOKUPTABLE = "/home/xinyu/Awesome_Database/Normative_modeling/Few_shot_brain/table/Brain_lookuptable.csv"
INFANT_DEFAULT_LOOKUPTABLE = "/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg4infaint/table/Brain_lookuptable.csv"
DEFAULT_LOOKUPTABLE = ADULT_DEFAULT_LOOKUPTABLE


@dataclass(frozen=True)
class LabelRule:
    label: int
    name: str
    max_cc: int | None


@dataclass(frozen=True)
class LabelGroups:
    rules: dict[int, LabelRule]
    wm_or_cortex: set[int]
    csf_ventricle_choroid: set[int]


def resolve_lookuptable_csv(lookuptable_csv: str | None = None, args_json: str | None = None, *paths: str | None) -> str:
    if lookuptable_csv is not None and str(lookuptable_csv).strip() != "":
        return str(lookuptable_csv)
    if args_json is not None and str(args_json).strip() != "":
        with open(args_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        candidate = cfg.get("left_right_pairs_csv", None)
        if candidate is not None and str(candidate).strip() != "":
            return str(candidate)
    joined = " ".join(str(p) for p in paths if p is not None)
    if "GG_Bond_Seg4infaint" in joined and Path(INFANT_DEFAULT_LOOKUPTABLE).exists():
        return INFANT_DEFAULT_LOOKUPTABLE
    return ADULT_DEFAULT_LOOKUPTABLE


def read_label_groups(lookuptable_csv: str | None = None, args_json: str | None = None) -> LabelGroups:
    lookuptable_csv = resolve_lookuptable_csv(lookuptable_csv, args_json)
    rules: dict[int, LabelRule] = {}
    wm_or_cortex: set[int] = set()
    csf_ventricle_choroid: set[int] = set()
    with open(lookuptable_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_text = (row.get("GGBond") or "").strip()
            if label_text == "":
                continue
            label = int(float(label_text))
            name = (row.get("Structure") or "").strip()
            cc_text = (row.get("CC") or "").strip()
            max_cc = int(float(cc_text)) if cc_text != "" else None
            rules[label] = LabelRule(label=label, name=name, max_cc=max_cc)

            lower_name = name.lower()
            if "cerebral white matter" in lower_name or "cerebral cortex" in lower_name:
                wm_or_cortex.add(label)
            if "csf" in lower_name or "ventricle" in lower_name or "choroid plexus" in lower_name:
                csf_ventricle_choroid.add(label)
    return LabelGroups(rules=rules, wm_or_cortex=wm_or_cortex, csf_ventricle_choroid=csf_ventricle_choroid)


def as_label_array(seg) -> np.ndarray:
    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    arr = np.asarray(seg)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def component_sizes(labeled: np.ndarray) -> np.ndarray:
    sizes = np.bincount(labeled.ravel())
    if sizes.size > 0:
        sizes[0] = 0
    return sizes


def keep_component_ids(labeled: np.ndarray, max_cc: int) -> set[int]:
    sizes = component_sizes(labeled)
    if sizes.size <= 1:
        return set()
    order = np.argsort(sizes)[::-1]
    return {int(i) for i in order[: max(1, int(max_cc))] if sizes[i] > 0}


def expand_slice(sl: tuple[slice, ...], shape: tuple[int, ...], margin: int = 1) -> tuple[slice, ...]:
    return tuple(slice(max(0, s.start - margin), min(shape[i], s.stop + margin)) for i, s in enumerate(sl))


def majority_vote(labels: np.ndarray, current_label: int, allowed: set[int] | None = None) -> int | None:
    labels = labels.astype(np.int64, copy=False)
    keep = (labels != 100) & (labels != current_label)
    if allowed is not None:
        keep &= np.isin(labels, list(allowed))
    labels = labels[keep]
    if labels.size == 0:
        return None
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def relabel_small_components_brainparc_style(
    seg: np.ndarray,
    groups: LabelGroups,
    connectivity: int = 1,
    labels_to_process: set[int] | None = None,
    restrict_wm_or_cortex: bool = True,
) -> np.ndarray:
    out = seg.copy()
    structure = ndimage.generate_binary_structure(seg.ndim, connectivity)

    for label in sorted(int(i) for i in np.unique(seg)):
        if label == 0:
            continue
        if labels_to_process is not None and label not in labels_to_process:
            continue
        rule = groups.rules.get(label)
        if rule is None or rule.max_cc is None:
            continue

        mask = out == label
        labeled, n_cc = ndimage.label(mask, structure=structure)
        if n_cc <= rule.max_cc:
            continue

        keep_ids = keep_component_ids(labeled, rule.max_cc)
        if not keep_ids:
            continue

        allowed = groups.csf_ventricle_choroid if restrict_wm_or_cortex and label in groups.wm_or_cortex else None
        objects = ndimage.find_objects(labeled)
        for comp_id, comp_slice in enumerate(objects, start=1):
            if comp_slice is None or comp_id in keep_ids:
                continue

            local_slice = expand_slice(comp_slice, seg.shape, margin=1)
            local_component = labeled[local_slice] == comp_id
            if not np.any(local_component):
                continue

            local_dilated = ndimage.binary_dilation(local_component, structure=structure, iterations=1)
            local_ring = local_dilated & ~local_component
            if not np.any(local_ring):
                continue

            new_label = majority_vote(out[local_slice][local_ring], current_label=label, allowed=allowed)
            if new_label is None:
                continue
            out[local_slice][local_component] = new_label
    return out


def post_process_segmentation_array(
    seg: np.ndarray,
    lookuptable_csv: str | None = None,
    args_json: str | None = None,
    max_radius: int = 3,
    connectivity: int = 1,
) -> np.ndarray:
    # max_radius is kept for API compatibility with inference_by_folder/csv.
    seg = as_label_array(seg)
    groups = read_label_groups(lookuptable_csv, args_json)
    out = relabel_small_components_brainparc_style(seg, groups, connectivity=connectivity)
    return relabel_small_components_brainparc_style(
        out,
        groups,
        connectivity=connectivity,
        labels_to_process=groups.wm_or_cortex,
        restrict_wm_or_cortex=False,
    )


def post_process_nii(
    seg_nii: str,
    output_nii: str,
    lookuptable_csv: str | None = None,
    args_json: str | None = None,
    max_radius: int = 3,
    connectivity: int = 1,
) -> str:
    # max_radius is kept for CLI compatibility.
    lookuptable_csv = resolve_lookuptable_csv(lookuptable_csv, args_json, seg_nii, output_nii)
    seg_img = nib.load(seg_nii)
    seg = as_label_array(seg_img.get_fdata())
    out = post_process_segmentation_array(
        seg,
        lookuptable_csv=lookuptable_csv,
        max_radius=max_radius,
        connectivity=connectivity,
    )
    output_path = Path(output_nii)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(np.int16), seg_img.affine, seg_img.header), str(output_path))
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lookuptable_csv", default=None, help="Brain_lookuptable.csv for post-processing. Overrides --args_json.")
    parser.add_argument("--args_json", default=None, help="Training args.json; uses left_right_pairs_csv when --lookuptable_csv is omitted.")
    parser.add_argument("--max_radius", type=int, default=3)
    parser.add_argument("--connectivity", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    post_process_nii(
        seg_nii=args.seg,
        output_nii=args.out,
        lookuptable_csv=args.lookuptable_csv,
        args_json=args.args_json,
        max_radius=args.max_radius,
        connectivity=args.connectivity,
    )


if __name__ == "__main__":
    main()
