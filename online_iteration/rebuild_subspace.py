from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.ndimage import zoom
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_PY = Path(__file__).resolve().parent / "confidence.py"
DEFAULT_TEMPLATE = (
    "/share/projectdata/Neural_Studio_Task/Code/xinyu/Structure_MRI/"
    "MNI152_template_T1_1mm_withoutSkull_intersubject/MNI152_T1.nii"
)
DEFAULT_IMAGE_REL = "step_1_T1w_process_ANTs-2.4.0_synthmorph/T1w2MNI_RigidWarped.nii.gz"
FALLBACK_IMAGE_REL = "step_1_T1w_process_ANTs-2.4.0/T1w2MNI_RigidWarped.nii.gz"


def run_cmd(cmd: str):
    print(f"[CMD] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def q(path) -> str:
    return shlex.quote(str(path))


def resolve_image_path(process_dir: str) -> str:
    image = os.path.join(str(process_dir), DEFAULT_IMAGE_REL)
    if not os.path.exists(image):
        image = os.path.join(str(process_dir), FALLBACK_IMAGE_REL)
    if not os.path.exists(image):
        raise FileNotFoundError(f"Cannot find processed image under: {process_dir}")
    return image


def subject_id(row) -> str:
    return f"{row['Site']}_{row['SubjectID']}_{row['Session']}"


def subject_parts(row):
    return str(row["Site"]), str(row["SubjectID"]), str(row["Session"])


def parse_labels(value) -> list[int]:
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, float) and not np.isnan(value):
        return [int(value)]
    text = str(value)
    nums = re.findall(r"-?\d+", text)
    if not nums:
        raise ValueError(f"Cannot parse label list from: {value!r}")
    return [int(i) for i in nums]


def resize_3d_to_shape(arr: np.ndarray, target_shape, order: int = 1) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    target_shape = tuple(int(x) for x in target_shape)
    if tuple(arr.shape) == target_shape:
        return arr.astype(np.float32, copy=False)
    factors = [target_shape[i] / arr.shape[i] for i in range(3)]
    out = zoom(arr, zoom=factors, order=order, mode="nearest", prefilter=False).astype(np.float32)
    out = out[: target_shape[0], : target_shape[1], : target_shape[2]]
    pad = [(0, max(0, target_shape[i] - out.shape[i])) for i in range(3)]
    if any(p[1] > 0 for p in pad):
        out = np.pad(out, pad, mode="edge")
    return out.astype(np.float32, copy=False)


def reconstruct_no_mean(mask_low: np.ndarray, v: np.ndarray, rank: int) -> np.ndarray:
    low_shape = tuple(mask_low.shape)
    x = mask_low.reshape(1, -1).astype(np.float32)
    vr = v[:rank].astype(np.float32)
    recon = (x @ vr.T) @ vr
    return np.clip(recon.reshape(low_shape), 0.0, 1.0).astype(np.float32)


def moving_average(x, window: int = 7):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size <= 2:
        return x
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = min(window, x.size if x.size % 2 == 1 else x.size - 1)
    if window <= 1:
        return x
    pad = window // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x_pad, kernel, mode="valid")


def find_elbow_by_chord(error_list, start_rank: int = 1, smooth_window: int = 7):
    err_raw = np.asarray(error_list, dtype=np.float64)
    if err_raw.size <= 1:
        return start_rank, 0, err_raw, err_raw, np.zeros_like(err_raw)
    err = moving_average(err_raw, window=smooth_window)
    ranks = np.arange(start_rank, start_rank + len(err))
    x = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)
    y = (err - err.min()) / (err.max() - err.min() + 1e-12)
    points = np.stack([x, y], axis=1)
    p1, p2 = points[0], points[-1]
    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec) + 1e-12
    distances = np.abs(np.cross(line_vec, points - p1)) / line_norm
    elbow_idx = int(np.argmax(distances))
    elbow_rank = int(ranks[elbow_idx])
    return elbow_rank, elbow_idx, err_raw, err, distances


def apply_transform(
    *,
    warp_path: str,
    save_path: str | Path,
    apply: str,
    apply_input: str | Path,
    apply_prefix: str,
    interpolation: str,
    template: str,
):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    out_path = save_path / f"{apply_prefix}2mni{apply}warped.nii.gz"
    if out_path.exists():
        return str(out_path)

    warp_path = Path(warp_path)
    synth_dir = warp_path / "step_1_T1w_process_ANTs-2.4.0_synthmorph"
    joint_warp = synth_dir / "T1w2mni_warp.mgz"
    joint_inv = synth_dir / "T1w2mni_inwarp.mgz"
    synth_interp = "nearest" if interpolation == "nearest" else "linear"
    if apply not in {"warp", "inwarp"}:
        raise ValueError(f"Unknown apply mode: {apply}")
    transform = joint_warp if apply == "warp" else joint_inv
    if not transform.exists():
        raise FileNotFoundError(
            f"SynthMorph transform not found: {transform}. "
            "This GitHub online iteration requires SynthMorph transforms."
        )
    cmd = f"mri_synthmorph apply -m {synth_interp} {q(transform)} {q(apply_input)} {q(out_path)}"

    run_cmd(cmd)
    return str(out_path)


def infer_bbox_json(prior_dir: str | None, tissue_csv: str, explicit: str | None) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if prior_dir:
        p = Path(prior_dir)
        candidates.append(p.parent / "mean_template" / "cut" / "bbox.json")
    csv = Path(tissue_csv)
    candidates.append(csv.parent.parent / "subspace" / "mean_template" / "cut" / "bbox.json")
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("Cannot find bbox.json. Tried: " + ", ".join(str(c) for c in candidates))


def load_tissue_table(tissue_csv: str):
    df = pd.read_csv(tissue_csv, encoding="utf-8-sig", usecols=lambda x: x != "Unnamed: 0")
    if "tissue" not in df.columns or "label" not in df.columns:
        raise ValueError(f"tissue csv must contain columns tissue,label: {tissue_csv}")
    df = df.copy()
    df["labels_parsed"] = df["label"].apply(parse_labels)
    return df


def warp_predictions_to_mni(args, df: pd.DataFrame):
    warped_root = Path(args.output_root) / "subspace" / "initial_segmentation"
    for _, row in tqdm(df.iterrows(), total=len(df), desc="warp pseudo labels to MNI"):
        site, sid, ses = subject_parts(row)
        subj = subject_id(row)
        pred = Path(args.output_root) / "predictions" / site / sid / ses / f"{subj}_ggbond_post.nii.gz"
        if not pred.exists():
            raise FileNotFoundError(f"Missing post-processed pseudo label: {pred}")
        save_dir = warped_root / site / sid / ses
        apply_transform(
            warp_path=str(row["process"]),
            save_path=save_dir,
            apply="warp",
            apply_input=pred,
            apply_prefix="brainseg",
            interpolation="nearest",
            template=args.template,
        )


def cut_for_subspace(args, df: pd.DataFrame, tissue_df: pd.DataFrame, bbox: dict):
    warped_root = Path(args.output_root) / "subspace" / "initial_segmentation"
    cut_root = Path(args.output_root) / "subspace" / "cut4subspace"
    for _, row in tqdm(df.iterrows(), total=len(df), desc="cut masks for subspace"):
        site, sid, ses = subject_parts(row)
        warped = warped_root / site / sid / ses / "brainseg2mniwarpwarped.nii.gz"
        if not warped.exists():
            raise FileNotFoundError(f"Missing warped pseudo label: {warped}")
        img = nib.load(str(warped))
        label = np.rint(img.get_fdata()).astype(np.int16)
        for _, trow in tissue_df.iterrows():
            tissue = str(trow["tissue"])
            x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox[tissue]]
            mask = np.isin(label, trow["labels_parsed"]).astype(np.float32)
            out_dir = cut_root / site / sid / ses / "nii" / tissue
            out_dir.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(mask, img.affine, img.header),
                str(out_dir / f"{tissue}.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(mask[x0:x1, y0:y1, z0:z1], np.eye(4)),
                str(out_dir / f"{tissue}_cut.nii.gz"),
            )


def make_subspace(args, df: pd.DataFrame, tissue_df: pd.DataFrame, bbox: dict):
    cut_root = Path(args.output_root) / "subspace" / "cut4subspace"
    subspace_root = Path(args.output_root) / "subspace" / "subspace_save"
    subspace_root.mkdir(parents=True, exist_ok=True)
    n_cases = len(df)

    for _, trow in tqdm(tissue_df.iterrows(), total=len(tissue_df), desc="make subspace"):
        tissue = str(trow["tissue"])
        save_dir = subspace_root / tissue
        save_dir.mkdir(parents=True, exist_ok=True)
        x0, x1, y0, y1, z0, z1 = [int(v) for v in bbox[tissue]]
        sx, sy, sz = x1 - x0, y1 - y0, z1 - z0
        if sx * sy * sz >= args.downsample_voxel_threshold:
            sx, sy, sz = max(1, sx // 2), max(1, sy // 2), max(1, sz // 2)
        low_shape = (sx, sy, sz)

        total = np.zeros((n_cases, sx, sy, sz), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            site, sid, ses = subject_parts(row)
            nii_path = cut_root / site / sid / ses / "nii" / tissue / f"{tissue}_cut.nii.gz"
            if not nii_path.exists():
                raise FileNotFoundError(f"Missing cut mask: {nii_path}")
            mask = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
            mask = resize_3d_to_shape(mask, low_shape, order=1)
            total[i] = (mask > 0.5).astype(np.float32)

        matrix = total.reshape(n_cases, -1)
        max_rank = min(int(args.subspace_rank), n_cases - 1, matrix.shape[1] - 1)
        if max_rank >= 1:
            u, s, v = sparse.linalg.svds(matrix + 1e-10, k=max_rank)
            order = np.argsort(s)[::-1]
            u, s, v = u[:, order], s[order], v[order]
            used_rank = int(max_rank)
        else:
            norm = np.linalg.norm(matrix[0]) if n_cases > 0 else 0.0
            v = matrix[:1] / max(norm, 1e-8)
            u = np.ones((n_cases, 1), dtype=np.float32)
            s = np.array([norm], dtype=np.float32)
            used_rank = 1

        save_file = save_dir / f"{tissue}_subspace_rank_{used_rank}.pt"
        torch.save(
            {
                "U": u.astype(np.float32),
                "S": s.astype(np.float32),
                "V": v.astype(np.float32),
                "shape": low_shape,
                "bbox": (x0, x1, y0, y1, z0, z1),
            },
            save_file,
        )
        error_list = []
        denom = np.linalg.norm(matrix, "fro") + 1e-12
        for r in range(1, used_rank + 1):
            vr = v[:r].astype(np.float32)
            recon = (matrix @ vr.T) @ vr
            error_list.append(float(np.linalg.norm(matrix - recon, "fro") / denom))
        elbow_rank, elbow_idx, err_raw, err_smooth, distances = find_elbow_by_chord(
            error_list,
            start_rank=1,
            smooth_window=7,
        )
        with open(save_dir / f"{tissue}_rank_error_curve.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tissue": tissue,
                    "elbow_rank": int(elbow_rank),
                    "rank": int(used_rank),
                    "self_reconstruction_error_list": [float(x) for x in err_raw],
                    "self_reconstruction_error_smooth": [float(x) for x in err_smooth],
                    "elbow_distance_to_chord": [float(x) for x in distances],
                    "note": "Elbow is computed on current-iteration pseudo labels because no separate train/test split is used online.",
                },
                f,
                indent=2,
            )


def reconstruct_priors(args, df: pd.DataFrame, tissue_df: pd.DataFrame, bbox: dict):
    template_img = nib.load(args.template)
    full_shape = template_img.shape[:3]
    template_affine = template_img.affine
    template_header = template_img.header.copy()
    template_header.set_data_dtype(np.float32)

    cut_root = Path(args.output_root) / "subspace" / "cut4subspace"
    subspace_root = Path(args.output_root) / "subspace" / "subspace_save"
    recon_root = Path(args.output_root) / "subspace" / "recon_save"
    recon_root.mkdir(parents=True, exist_ok=True)

    for _, trow in tqdm(tissue_df.iterrows(), total=len(tissue_df), desc="reconstruct priors"):
        tissue = str(trow["tissue"])
        subspace_dir = subspace_root / tissue
        files = sorted(subspace_dir.glob(f"{tissue}_subspace_rank_*.pt"))
        if not files:
            raise FileNotFoundError(f"No subspace file for tissue {tissue}: {subspace_dir}")
        subspace_file = files[-1]
        subspace = torch.load(str(subspace_file), map_location="cpu")
        v = np.asarray(subspace["V"], dtype=np.float32)
        low_shape = tuple(int(i) for i in subspace["shape"])
        rank_json = subspace_dir / f"{tissue}_rank_error_curve.json"
        if args.recon_rank is not None:
            rank = int(args.recon_rank)
        elif rank_json.exists():
            with open(rank_json, "r", encoding="utf-8") as f:
                rank = int(json.load(f).get("elbow_rank", v.shape[0]))
        else:
            rank = v.shape[0]
        rank = max(1, min(v.shape[0], rank))
        x0, x1, y0, y1, z0, z1 = [int(vv) for vv in bbox[tissue]]
        bbox_shape = (x1 - x0, y1 - y0, z1 - z0)

        for _, row in df.iterrows():
            site, sid, ses = subject_parts(row)
            subj = subject_id(row)
            cut_mask = cut_root / site / sid / ses / "nii" / tissue / f"{tissue}_cut.nii.gz"
            mask = nib.load(str(cut_mask)).get_fdata(dtype=np.float32)
            mask_low = resize_3d_to_shape(mask, low_shape, order=1)
            mask_low = (mask_low > 0.5).astype(np.float32)
            recon_low = reconstruct_no_mean(mask_low, v, rank)
            recon_bbox = resize_3d_to_shape(recon_low, bbox_shape, order=1)
            full = np.zeros(full_shape, dtype=np.float32)
            full[x0:x1, y0:y1, z0:z1] = np.clip(recon_bbox, 0.0, 1.0)
            out_dir = recon_root / subj
            out_dir.mkdir(parents=True, exist_ok=True)
            out_mni = out_dir / f"{tissue}_recon.nii.gz"
            nib.save(nib.Nifti1Image(full, template_affine, template_header.copy()), str(out_mni))

            apply_transform(
                warp_path=str(row["process"]),
                save_path=out_dir,
                apply="inwarp",
                apply_input=out_mni,
                apply_prefix=f"{tissue}_recon",
                interpolation="linear",
                template=args.template,
            )


def run_confidence(args, df: pd.DataFrame):
    confidence_root = Path(args.output_root) / "confidence_save"
    recon_root = Path(args.output_root) / "subspace" / "recon_save"
    confidence_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="run confidence"):
        site, sid, ses = subject_parts(row)
        subj = subject_id(row)
        image = resolve_image_path(str(row["process"]))
        label = Path(args.output_root) / "predictions" / site / sid / ses / f"{subj}_ggbond_post.nii.gz"
        out_dir = confidence_root / subj
        prior_dir = recon_root / subj
        final = out_dir / "npy" / "posterior_q.npy"
        if final.exists() and not args.force:
            continue
        cmd = (
            f"{q(sys.executable)} {q(CONFIDENCE_PY)} "
            f"--out_dir {q(out_dir)} "
            f"--image {q(image)} "
            f"--label {q(label)} "
            f"--csv_path {q(args.tissue_csv)} "
            f"--prior_dir {q(prior_dir)} "
        )
        if args.force:
            cmd += " --force"
        run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tissue_csv", required=True)
    parser.add_argument("--prior_dir", default=None, help="Only used to infer the prepared bbox path if --bbox_json is not given.")
    parser.add_argument("--bbox_json", default=None)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--subspace_rank", type=int, default=1000)
    parser.add_argument("--recon_rank", type=int, default=None)
    parser.add_argument("--downsample_voxel_threshold", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig", usecols=lambda x: x != "Unnamed: 0")
    if "process" not in df.columns:
        raise RuntimeError("input_csv must contain a process column for subspace reconstruction")
    tissue_df = load_tissue_table(args.tissue_csv)
    bbox_json = infer_bbox_json(args.prior_dir, args.tissue_csv, args.bbox_json)
    with open(bbox_json, "r", encoding="utf-8") as f:
        bbox = json.load(f)
    missing = [str(t) for t in tissue_df["tissue"].tolist() if str(t) not in bbox]
    if missing:
        raise KeyError(f"bbox.json is missing tissues: {missing}")

    print(f"[subspace] output_root = {args.output_root}")
    print(f"[subspace] bbox_json   = {bbox_json}")
    warp_predictions_to_mni(args, df)
    cut_for_subspace(args, df, tissue_df, bbox)
    make_subspace(args, df, tissue_df, bbox)
    reconstruct_priors(args, df, tissue_df, bbox)
    run_confidence(args, df)
    print(f"[subspace] confidence_save = {Path(args.output_root) / 'confidence_save'}")


if __name__ == "__main__":
    main()
