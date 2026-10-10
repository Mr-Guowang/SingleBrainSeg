#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# Basic IO
# ============================================================

def load_nii(path, dtype=np.float32):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=dtype)
    return img, data


def load_label_nii(path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    data = np.rint(data).astype(np.int16)
    return img, data


def save_nii_like(data, ref_img, out_path, dtype=None):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        data = data.astype(dtype)

    new_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
    if dtype is not None:
        new_img.set_data_dtype(dtype)

    nib.save(new_img, out_path)


def copy_raw_image_once(src_path, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[SKIP] image_raw exists: {out_path}")
        return
    shutil.copy2(str(src_path), str(out_path))
    print(f"[COPY] image_raw -> {out_path}")


def save_json(obj, out_path):
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Tissue table
# ============================================================

def parse_label_cell(x):
    """
    Parse label cell:
        "6 21"
        "8 9 23 24"
        "12"
    """
    if pd.isna(x):
        return []

    s = str(x).strip()
    for sep in [",", ";", "|", "/"]:
        s = s.replace(sep, " ")

    labs = []
    for p in s.split():
        if p.strip() == "":
            continue
        labs.append(int(float(p)))

    return labs


def load_tissue_csv(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "tissue" not in df.columns:
        raise ValueError("csv must contain column: tissue")
    if "label" not in df.columns:
        raise ValueError("csv must contain column: label")

    tissues = []
    label_groups = []
    subspace_types = []

    for _, row in df.iterrows():
        tissue = str(row["tissue"]).strip()
        labels = parse_label_cell(row["label"])

        if tissue == "" or len(labels) == 0:
            continue

        if "subspace" in df.columns:
            subspace = str(row["subspace"]).strip()
        else:
            subspace = ""

        tissues.append(tissue)
        label_groups.append(labels)
        subspace_types.append(subspace)

    if len(tissues) == 0:
        raise RuntimeError("No valid tissue rows found in csv.")

    return tissues, label_groups, subspace_types


def build_merged_label(raw_label, tissues, label_groups):
    """
    raw_label: original label map.
    merged_idx: tissue index map.
        -1 means unmatched.
        0 means tissues[0], 1 means tissues[1], ...
    """
    merged_idx = np.full(raw_label.shape, -1, dtype=np.int16)

    label_to_tissue_idx = {}
    label_to_tissue_name = {}

    for k, (tissue, labs) in enumerate(zip(tissues, label_groups)):
        for lab in labs:
            if lab in label_to_tissue_idx:
                old = label_to_tissue_name[lab]
                raise ValueError(
                    f"Label {lab} appears in multiple tissues: {old} and {tissue}"
                )

            label_to_tissue_idx[lab] = k
            label_to_tissue_name[lab] = tissue
            merged_idx[raw_label == lab] = k

    return merged_idx, label_to_tissue_idx, label_to_tissue_name


# ============================================================
# Z-score image
# ============================================================

def zscore_2_98(image, mode="nonzero", raw_label=None):
    """
    Compute mean/std from intensities within 2-98 percentile range.

    mode:
        nonzero       : finite and image != 0
        label_nonzero : finite and raw_label != 0
        all           : all finite voxels
    """
    image = image.astype(np.float32)
    finite = np.isfinite(image)

    if mode == "nonzero":
        mask = finite & (image != 0)
    elif mode == "label_nonzero":
        if raw_label is None:
            raise ValueError("raw_label is required for mode='label_nonzero'")
        mask = finite & (raw_label != 0)
    elif mode == "all":
        mask = finite
    else:
        raise ValueError(f"Unknown zscore mode: {mode}")

    vals = image[mask]
    if vals.size < 100:
        raise RuntimeError(
            f"Too few voxels for z-score estimation: {vals.size}. "
            f"Try --zscore_mode all"
        )

    p2, p98 = np.percentile(vals, [2, 98])
    stat_mask = mask & (image >= p2) & (image <= p98)
    stat_vals = image[stat_mask]

    mean = float(np.mean(stat_vals))
    std = float(np.std(stat_vals))

    if std < 1e-8:
        raise RuntimeError(f"std too small: {std}")

    zimg = (image - mean) / std
    zimg[~finite] = 0
    zimg = zimg.astype(np.float32)

    info = {
        "zscore_mode": mode,
        "p2": float(p2),
        "p98": float(p98),
        "mean_2_98": mean,
        "std_2_98": std,
        "num_voxels_for_percentile": int(vals.size),
        "num_voxels_for_mean_std": int(stat_vals.size),
    }

    return zimg, info


# ============================================================
# Prior
# ============================================================

def load_and_normalize_priors(prior_dir, tissues, shape, suffix, eps=1e-8):
    """
    Load prior maps:
        {tissue}{suffix}

    Then normalize over tissue dimension:
        pi_k(x) = (prior_k(x)+eps) / sum_j(prior_j(x)+eps)
    """
    prior_list = []

    for idx,tissue in enumerate(tissues):
        prior_path = os.path.join(prior_dir, f"{tissue}{suffix}")

        if not os.path.exists(prior_path):
            raise FileNotFoundError(f"Missing prior: {prior_path}")

        _, prior = load_nii(prior_path, dtype=np.float32)

        if prior.shape != shape:
            raise ValueError(
                f"Prior shape mismatch for {tissue}:\n"
                f"prior shape = {prior.shape}\n"
                f"expected    = {shape}\n"
                f"path        = {prior_path}"
            )

        prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        prior[prior < 0] = 0
        if tissue == 'bg':
            prior_copy = prior.copy()
            index = idx
        prior_list.append(prior.astype(np.float32))

    prior_raw = np.stack(prior_list, axis=0)  # K, X, Y, Z
    prior_sum = np.sum(prior_raw, axis=0, keepdims=True)
    print('################## sum shape', prior_sum.shape)
    prior_copy[prior_sum[0,:,:,:] == 0] = 0.0001
    prior_raw[index,:,:,:] =  prior_copy
    prior_sum = np.sum(prior_raw, axis=0, keepdims=True)
    prior_norm = (prior_raw + eps) / (prior_sum + eps * len(tissues))
    prior_norm = prior_norm.astype(np.float32)

    return prior_norm


# ============================================================
# Likelihood
# ============================================================

def estimate_gaussian_likelihood_params(
    zimg,
    merged_idx,
    tissues,
    min_sigma=0.05,
    shrink_tau=0.0,
):
    """
    Estimate one Gaussian likelihood for each merged tissue:
        L_k(I_x) = N(I_x ; mu_k, sigma_k^2)
    """
    params = []

    for k, tissue in enumerate(tissues):
        mask = merged_idx == k
        vals = zimg[mask]
        vals = vals[np.isfinite(vals)]

        n = int(vals.size)

        if n == 0:
            mu_raw = 0.0
            sigma_raw = 1.0
        else:
            mu_raw = float(np.mean(vals))
            sigma_raw = float(np.std(vals))

        sigma_raw = max(sigma_raw, min_sigma)

        if shrink_tau > 0:
            lam = n / (n + shrink_tau)
            mu = lam * mu_raw + (1.0 - lam) * 0.0
            sigma2 = lam * sigma_raw ** 2 + (1.0 - lam) * 1.0
            sigma = float(np.sqrt(max(sigma2, min_sigma ** 2)))
        else:
            lam = 1.0
            mu = mu_raw
            sigma = sigma_raw

        params.append({
            "tissue": tissue,
            "n_voxels": n,
            "mu_raw": float(mu_raw),
            "sigma_raw": float(sigma_raw),
            "shrink_lambda": float(lam),
            "mu": float(mu),
            "sigma": float(sigma),
        })

    return params


def compute_log_likelihood_stack(zimg, likelihood_params):
    """
    Return log likelihood:
        logL[k, x] = log N(I_x ; mu_k, sigma_k^2)
    """
    K = len(likelihood_params)
    shape = zimg.shape
    logL = np.zeros((K,) + shape, dtype=np.float32)

    const = -0.5 * np.log(2.0 * np.pi)

    for k, p in enumerate(likelihood_params):
        mu = float(p["mu"])
        sigma = max(float(p["sigma"]), 1e-8)

        logL[k] = (
            -0.5 * ((zimg - mu) / sigma) ** 2
            - np.log(sigma)
            + const
        ).astype(np.float32)

    return logL


def softmax_from_log(log_score, eps=1e-8):
    """
    Stable softmax over axis=0.
    """
    m = np.max(log_score, axis=0, keepdims=True)
    exp_score = np.exp(log_score - m).astype(np.float32)
    denom = np.sum(exp_score, axis=0, keepdims=True) + eps
    return (exp_score / denom).astype(np.float32)


# ============================================================
# Posterior, confidence, uncertainty
# ============================================================

def compute_posterior(prior_norm, logL, alpha_prior=1.0, eps=1e-8):
    """
    Bayesian fusion:
        q_k(x) ∝ L_k(I_x) * pi_k(x)^alpha
    """
    log_prior = np.log(prior_norm + eps).astype(np.float32)
    log_score = logL + alpha_prior * log_prior
    posterior = softmax_from_log(log_score, eps=eps)
    return posterior


def compute_uncertainty_and_certainty(posterior, eps=1e-8):
    """
    Global posterior uncertainty:
        U(x) = H(q(x)) / log(K)

    Certainty:
        G(x) = 1 - U(x)
    """
    K = posterior.shape[0]

    entropy = -np.sum(
        posterior * np.log(posterior + eps),
        axis=0
    ).astype(np.float32)

    if K > 1:
        uncertainty = entropy / np.log(K)
    else:
        uncertainty = np.zeros_like(entropy, dtype=np.float32)

    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32)
    certainty = (1.0 - uncertainty).astype(np.float32)

    return uncertainty, certainty, entropy


def compute_label_confidence(posterior, certainty, merged_idx):
    """
    Local label confidence based on original label's merged tissue.

    If raw label x belongs to tissue k:
        confidence(x) = posterior[k, x] * certainty(x)

    If raw label is unmatched:
        confidence(x) = 0
        assigned_q(x) = 0
    """
    shape = merged_idx.shape
    K = posterior.shape[0]

    assigned_q = np.zeros(shape, dtype=np.float32)
    confidence = np.zeros(shape, dtype=np.float32)

    for k in range(K):
        mask = merged_idx == k
        if not np.any(mask):
            continue

        qk = posterior[k]
        assigned_q[mask] = qk[mask]
        confidence[mask] = qk[mask] * certainty[mask]

    return assigned_q, confidence


# ============================================================
# Save outputs
# ============================================================

def save_tissue_maps(
    arr4d,
    tissues,
    ref_img,
    out_dir,
    prefix,
    dtype=np.float32,
):
    """
    Save arr4d[k] as individual nii.gz maps.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, tissue in enumerate(tissues):
        out_path = out_dir / f"{k:02d}_{tissue}_{prefix}.nii.gz"
        save_nii_like(arr4d[k], ref_img, out_path, dtype=dtype)


def summarize_by_label_and_tissue(
    raw_label,
    merged_idx,
    tissues,
    label_to_tissue_idx,
    label_to_tissue_name,
    assigned_q,
    confidence,
    uncertainty,
    certainty,
    out_dir,
):
    rows_label = []

    raw_labels = sorted(np.unique(raw_label).astype(int).tolist())

    for lab in raw_labels:
        mask = raw_label == lab
        n = int(np.sum(mask))

        if lab in label_to_tissue_idx:
            tidx = int(label_to_tissue_idx[lab])
            tissue = label_to_tissue_name[lab]
        else:
            tidx = -1
            tissue = "unmatched"

        rows_label.append({
            "label": int(lab),
            "tissue_index": tidx,
            "tissue": tissue,
            "n_voxels": n,
            "mean_assigned_q": float(np.mean(assigned_q[mask])) if n > 0 else np.nan,
            "mean_confidence_qG": float(np.mean(confidence[mask])) if n > 0 else np.nan,
            "mean_uncertainty": float(np.mean(uncertainty[mask])) if n > 0 else np.nan,
            "mean_certainty_G": float(np.mean(certainty[mask])) if n > 0 else np.nan,
        })

    df_label = pd.DataFrame(rows_label)
    df_label.to_csv(Path(out_dir) / "summary_by_original_label.csv", index=False)

    rows_tissue = []

    for k, tissue in enumerate(tissues):
        mask = merged_idx == k
        n = int(np.sum(mask))

        rows_tissue.append({
            "tissue_index": int(k),
            "tissue": tissue,
            "n_voxels": n,
            "mean_assigned_q": float(np.mean(assigned_q[mask])) if n > 0 else np.nan,
            "mean_confidence_qG": float(np.mean(confidence[mask])) if n > 0 else np.nan,
            "mean_uncertainty": float(np.mean(uncertainty[mask])) if n > 0 else np.nan,
            "mean_certainty_G": float(np.mean(certainty[mask])) if n > 0 else np.nan,
        })

    df_tissue = pd.DataFrame(rows_tissue)
    df_tissue.to_csv(Path(out_dir) / "summary_by_merged_tissue.csv", index=False)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--prior_dir", type=str, required=True)

    parser.add_argument(
        "--prior_suffix",
        type=str,
        default="_recon2mniinwarpwarped.nii.gz",
        help="Prior file name is {tissue}{prior_suffix}",
    )

    parser.add_argument(
        "--zscore_mode",
        type=str,
        default="nonzero",
        choices=["nonzero", "label_nonzero", "all"],
    )

    parser.add_argument(
        "--min_sigma",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--shrink_tau",
        type=float,
        default=0.0,
        help="If >0, shrink tissue Gaussian stats toward global N(0,1).",
    )

    parser.add_argument(
        "--alpha_prior",
        type=float,
        default=1.0,
        help="Posterior uses L_k * prior_k^alpha_prior.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--save_likelihood_maps",
        type=int,
        default=1,
        help="Save normalized likelihood evidence maps as nii.gz.",
    )

    parser.add_argument(
        "--save_prior_maps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--save_posterior_maps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs. image_raw.nii.gz is only copied if missing and ignores this flag.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nii_dir = out_dir / "nii"
    npy_dir = out_dir / "npy"

    nii_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    copy_raw_image_once(args.image, nii_dir / "image_raw.nii.gz")

    final_result = nii_dir / "assigned_label_confidence_qG.nii.gz"
    if final_result.exists() and not args.force:
        print(f"[SKIP] final result exists: {final_result}")
        print("       use --force to recompute and overwrite outputs")
        print(f"[OUT] {out_dir}")
        return

    print("[1] Load image and label")
    img_obj, image = load_nii(args.image, dtype=np.float32)
    label_obj, raw_label = load_label_nii(args.label)

    if image.shape != raw_label.shape:
        raise ValueError(
            f"image and label shape mismatch: {image.shape} vs {raw_label.shape}"
        )

    shape = image.shape

    print("[2] Load tissue CSV and merge labels")
    tissues, label_groups, subspace_types = load_tissue_csv(args.csv_path)

    merged_idx, label_to_tissue_idx, label_to_tissue_name = build_merged_label(
        raw_label=raw_label,
        tissues=tissues,
        label_groups=label_groups,
    )

    K = len(tissues)

    print(f"    K tissues = {K}")
    for k, (t, labs) in enumerate(zip(tissues, label_groups)):
        print(f"    {k:02d}: {t:20s} labels={labs}")

    print("[3] Z-score image using 2-98 percentile statistics")
    zimg, zinfo = zscore_2_98(
        image=image,
        mode=args.zscore_mode,
        raw_label=raw_label,
    )

    print("[4] Load and normalize priors")
    prior_norm = load_and_normalize_priors(
        prior_dir=args.prior_dir,
        tissues=tissues,
        shape=shape,
        suffix=args.prior_suffix,
        eps=args.eps,
    )

    print("[5] Estimate tissue likelihood")
    likelihood_params = estimate_gaussian_likelihood_params(
        zimg=zimg,
        merged_idx=merged_idx,
        tissues=tissues,
        min_sigma=args.min_sigma,
        shrink_tau=args.shrink_tau,
    )

    for p in likelihood_params:
        print(
            f"    {p['tissue']:20s} "
            f"n={p['n_voxels']:8d} "
            f"mu={p['mu']:+.4f} "
            f"sigma={p['sigma']:.4f}"
        )

    print("[6] Compute likelihood evidence")
    logL = compute_log_likelihood_stack(
        zimg=zimg,
        likelihood_params=likelihood_params,
    )

    # This is not used as the Bayesian posterior.
    # It is only a normalized visualization of intensity-only evidence.
    likelihood_norm = softmax_from_log(logL, eps=args.eps)

    print("[7] Compute Bayesian posterior")
    posterior = compute_posterior(
        prior_norm=prior_norm,
        logL=logL,
        alpha_prior=args.alpha_prior,
        eps=args.eps,
    )

    print("[8] Compute global uncertainty and certainty")
    uncertainty, certainty, entropy = compute_uncertainty_and_certainty(
        posterior=posterior,
        eps=args.eps,
    )

    print("[9] Compute original-label-based local confidence")
    assigned_q, label_confidence = compute_label_confidence(
        posterior=posterior,
        certainty=certainty,
        merged_idx=merged_idx,
    )

    print("[10] Save npy arrays")
    np.save(npy_dir / "image_zscore.npy", zimg.astype(np.float32))
    np.save(npy_dir / "label_raw.npy", raw_label.astype(np.int16))
    np.save(npy_dir / "label_merged_tissue_index.npy", merged_idx.astype(np.int16))

    np.save(npy_dir / "prior_normalized.npy", prior_norm.astype(np.float32))
    np.save(npy_dir / "likelihood_normalized.npy", likelihood_norm.astype(np.float32))
    np.save(npy_dir / "posterior_q.npy", posterior.astype(np.float32))

    np.save(npy_dir / "uncertainty_entropy_norm.npy", uncertainty.astype(np.float32))
    np.save(npy_dir / "certainty_G.npy", certainty.astype(np.float32))
    np.save(npy_dir / "assigned_label_posterior_q.npy", assigned_q.astype(np.float32))
    np.save(npy_dir / "assigned_label_confidence_qG.npy", label_confidence.astype(np.float32))

    print("[11] Save main nii.gz maps")
    save_nii_like(zimg, img_obj, nii_dir / "image_zscore.nii.gz", dtype=np.float32)
    save_nii_like(raw_label, img_obj, nii_dir / "label_raw.nii.gz", dtype=np.int16)
    save_nii_like(merged_idx, img_obj, nii_dir / "label_merged_tissue_index.nii.gz", dtype=np.int16)

    save_nii_like(uncertainty, img_obj, nii_dir / "uncertainty_entropy_norm.nii.gz", dtype=np.float32)
    save_nii_like(certainty, img_obj, nii_dir / "certainty_G.nii.gz", dtype=np.float32)
    save_nii_like(assigned_q, img_obj, nii_dir / "assigned_label_posterior_q.nii.gz", dtype=np.float32)
    save_nii_like(label_confidence, img_obj, nii_dir / "assigned_label_confidence_qG.nii.gz", dtype=np.float32)

    if args.save_prior_maps:
        print("[12] Save normalized prior nii.gz maps")
        save_tissue_maps(
            arr4d=prior_norm,
            tissues=tissues,
            ref_img=img_obj,
            out_dir=nii_dir / "prior_normalized",
            prefix="prior_norm",
            dtype=np.float32,
        )

    if args.save_likelihood_maps:
        print("[13] Save normalized likelihood nii.gz maps")
        save_tissue_maps(
            arr4d=likelihood_norm,
            tissues=tissues,
            ref_img=img_obj,
            out_dir=nii_dir / "likelihood_normalized",
            prefix="likelihood_norm",
            dtype=np.float32,
        )

    if args.save_posterior_maps:
        print("[14] Save posterior nii.gz maps")
        save_tissue_maps(
            arr4d=posterior,
            tissues=tissues,
            ref_img=img_obj,
            out_dir=nii_dir / "posterior_q",
            prefix="posterior_q",
            dtype=np.float32,
        )

    print("[15] Save summaries")
    summarize_by_label_and_tissue(
        raw_label=raw_label,
        merged_idx=merged_idx,
        tissues=tissues,
        label_to_tissue_idx=label_to_tissue_idx,
        label_to_tissue_name=label_to_tissue_name,
        assigned_q=assigned_q,
        confidence=label_confidence,
        uncertainty=uncertainty,
        certainty=certainty,
        out_dir=out_dir,
    )

    metadata = {
        "image": args.image,
        "label": args.label,
        "csv_path": args.csv_path,
        "prior_dir": args.prior_dir,
        "prior_suffix": args.prior_suffix,
        "shape": list(shape),
        "tissues": tissues,
        "label_groups": label_groups,
        "subspace_types": subspace_types,
        "label_to_tissue_index": {str(k): int(v) for k, v in label_to_tissue_idx.items()},
        "label_to_tissue_name": {str(k): str(v) for k, v in label_to_tissue_name.items()},
        "zscore_info": zinfo,
        "likelihood_params": likelihood_params,
        "alpha_prior": args.alpha_prior,
        "min_sigma": args.min_sigma,
        "shrink_tau": args.shrink_tau,
        "definition": {
            "prior": "pi_k(x), normalized over merged tissues",
            "likelihood": "L_k(I_x) = Gaussian likelihood estimated from z-scored image and merged labels",
            "posterior": "q_k(x) = L_k(I_x) * pi_k(x)^alpha / sum_j L_j(I_x) * pi_j(x)^alpha",
            "uncertainty": "U(x) = entropy(q(x)) / log(K), computed over all merged tissues",
            "certainty": "G(x) = 1 - U(x)",
            "assigned_label_confidence": "C(x) = q_{merged(y(x))}(x) * G(x), based on original label's merged tissue",
        },
        "outputs": {
            "npy_dir": str(npy_dir),
            "nii_dir": str(nii_dir),
            "main_maps": {
                "image_zscore": str(nii_dir / "image_zscore.nii.gz"),
                "label_raw": str(nii_dir / "label_raw.nii.gz"),
                "label_merged_tissue_index": str(nii_dir / "label_merged_tissue_index.nii.gz"),
                "uncertainty_entropy_norm": str(nii_dir / "uncertainty_entropy_norm.nii.gz"),
                "certainty_G": str(nii_dir / "certainty_G.nii.gz"),
                "assigned_label_posterior_q": str(nii_dir / "assigned_label_posterior_q.nii.gz"),
                "assigned_label_confidence_qG": str(nii_dir / "assigned_label_confidence_qG.nii.gz"),
            },
            "summary_by_original_label": str(out_dir / "summary_by_original_label.csv"),
            "summary_by_merged_tissue": str(out_dir / "summary_by_merged_tissue.csv"),
        },
    }

    save_json(metadata, out_dir / "metadata.json")

    print("[DONE]")
    print(f"[OUT] {out_dir}")


if __name__ == "__main__":
    main()