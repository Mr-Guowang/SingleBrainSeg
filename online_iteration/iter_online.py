from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKBONE_DIR = PROJECT_ROOT / "backbone"
DEFAULT_DATASET_JSON = str(PROJECT_ROOT / "lookuptable" / "dataset.json")
DEFAULT_BAYES_TISSUE_CSV = "/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/table/subspace_table.csv"
DEFAULT_PRIOR_DIR = "/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/subspace/confidence_save"
DEFAULT_IMAGE_REL = "step_1_T1w_process_ANTs-2.4.0_synthmorph/T1w2MNI_RigidWarped.nii.gz"
FALLBACK_IMAGE_REL = "step_1_T1w_process_ANTs-2.4.0/T1w2MNI_RigidWarped.nii.gz"
py = sys.executable
segpy = str(BACKBONE_DIR / "inference_by_csv.py")
postpy = str(BACKBONE_DIR / "post_process.py")
bayespy = str(Path(__file__).resolve().parent / "bayes_one.py")
rebuild_subspace_py = str(Path(__file__).resolve().parent / "rebuild_subspace.py")
batch_size = 8


def run_batch(cmd_list):
    for cmd in cmd_list:
        subprocess.run(cmd, shell=True, check=True)


def run_seg(args):
    spacing = " ".join(str(float(i)) for i in args.target_spacing)
    cmd = f'{py} {segpy} --args_json {args.args_json} --input_csv {args.input_csv} --output_folder {args.output_root}/predictions --checkpoint {args.checkpoint} --gpu {args.gpu} --target_spacing {spacing} '
    os.system(cmd)

def resolve_image_path(process_dir: str) -> str:
    image = os.path.join(str(process_dir), DEFAULT_IMAGE_REL)
    if not os.path.exists(image):
        image = os.path.join(str(process_dir), FALLBACK_IMAGE_REL)
    return image


def run_post(args):
    df = pd.read_csv(args.input_csv)
    cmd_list = []
    for index, row in tqdm(df[:].iterrows(), total=len(df[:]), desc="Processing Rows"):
        Site, SubjectID, Session = str(row['Site']), str(row['SubjectID']), str(row['Session'])
        subject = f'{Site}_{SubjectID}_{Session}'
        seg = os.path.join(args.output_root, 'predictions', Site, SubjectID, Session, f'{subject}_ggbond.nii.gz')
        seg_post = os.path.join(args.output_root, 'predictions', Site, SubjectID, Session, f'{subject}_ggbond_post.nii.gz')
        cmd_list.append(f'{py} {postpy} --seg {seg} --out {seg_post} --args_json {args.args_json}')
        if len(cmd_list) >= batch_size:
            run_batch(cmd_list)
            cmd_list = []
    run_batch(cmd_list)
    cmd_list = []


def run_iter_synth(args):
    if not args.update_synth:
        return
    df = pd.read_csv(args.input_csv)
    synth_dir = os.path.join(args.output_root, 'iter_synth')
    os.makedirs(synth_dir, exist_ok=True)
    for index, row in tqdm(df[:].iterrows(), total=len(df[:]), desc="Synth Label Rows"):
        Site, SubjectID, Session = str(row['Site']), str(row['SubjectID']), str(row['Session'])
        subject = f'{Site}_{SubjectID}_{Session}'
        seg_post = os.path.join(args.output_root, 'predictions', Site, SubjectID, Session, f'{subject}_ggbond_post.nii.gz')
        if not os.path.exists(seg_post):
            raise FileNotFoundError(f"Missing post label for iter_synth: {seg_post}")
        out_label = os.path.join(synth_dir, f'{subject}_synth.nii.gz')
        if args.force or not os.path.exists(out_label):
            shutil.copy2(seg_post, out_label)
    print(f"[iter_synth] updated synth labels: {synth_dir}")


def run_rebuild_subspace(args):
    cmd = (
        f'{py} {rebuild_subspace_py} '
        f'--input_csv {args.input_csv} '
        f'--output_root {args.output_root} '
        f'--tissue_csv {args.bayes_tissue_csv} '
    )
    if args.prior_dir:
        cmd += f'--prior_dir {args.prior_dir} '
    if args.subspace_bbox_json:
        cmd += f'--bbox_json {args.subspace_bbox_json} '
    if args.subspace_template:
        cmd += f'--template {args.subspace_template} '
    cmd += f'--subspace_rank {int(args.subspace_rank)} '
    if args.subspace_recon_rank is not None:
        cmd += f'--recon_rank {int(args.subspace_recon_rank)} '
    if args.force:
        cmd += '--force '
    run_batch([cmd])
    args.prior_dir = os.path.join(args.output_root, 'confidence_save')


def run_bayes(args):
    df = pd.read_csv(args.input_csv)
    cmd_list = []
    output_folder = args.processed_weakdir or os.path.join(args.output_root, 'weak_processed')
    prior_dir = args.prior_dir or DEFAULT_PRIOR_DIR
    spacing = " ".join(str(float(i)) for i in args.target_spacing)
    force = ' --overwrite' if args.force else ''

    for index, row in tqdm(df[:].iterrows(), total=len(df[:]), desc="Bayes Rows"):
        Site, SubjectID, Session = str(row['Site']), str(row['SubjectID']), str(row['Session'])
        subject = f'{Site}_{SubjectID}_{Session}'
        if 'process' not in row or pd.isna(row['process']):
            raise RuntimeError('input_csv must contain a valid process column for bayes training-data generation')
        image = resolve_image_path(str(row['process']))
        label = os.path.join(args.output_root, 'predictions', Site, SubjectID, Session, f'{subject}_ggbond_post.nii.gz')
        
        prior = os.path.join(prior_dir, subject)
        cmd_list.append(
            f'{py} {bayespy} '
            f'--image {image} '
            f'--label {label} '
            f'--prior {prior} '
            f'--output_folder {output_folder} '
            f'--subject {subject} '
            f'--dataset_json {args.dataset_json} '
            f'--tissue_csv {args.bayes_tissue_csv} '
            f'--target_spacing {spacing}'
            f'{force}'
        )
        if len(cmd_list) >= batch_size:
            run_batch(cmd_list)
            cmd_list = []
    run_batch(cmd_list)
    cmd_list = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--processed_weakdir", default=None)
    parser.add_argument("--dataset_json", default=DEFAULT_DATASET_JSON)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--bayes_tissue_csv", default=DEFAULT_BAYES_TISSUE_CSV)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--subspace_bbox_json", default=None)
    parser.add_argument("--subspace_template", default=None)
    parser.add_argument("--subspace_rank", type=int, default=1000)
    parser.add_argument("--subspace_recon_rank", type=int, default=None)
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update_synth", action="store_true", help="Copy post-processed pseudo labels to output_root/iter_synth for SynthSeg label generation.")
    args = parser.parse_args()
    run_seg(args)
    run_post(args)
    run_iter_synth(args)
    run_rebuild_subspace(args)
    run_bayes(args)

if __name__ == "__main__":
    main()
