# SingleBrainSeg

SingleBrainSeg is a brain MRI segmentation framework designed for robust anatomical segmentation with pseudo-label refinement and structural subspace guidance.

This repository is under active maintenance. We will continue to update the code, documentation, pretrained weights, and examples.

## Latest Updates

| Date | Update |
| --- | --- |
| 2026-10-10 | Initial GitHub release with Python testing code and test weights. |

## Notice

This release provides the testing/inference code and test weights. The complete cleaned training code will be released after the paper is officially accepted.

We are still polishing the repository for public use, so the interface may be updated over time. Please check this page for future updates.

## Download Test Weights

The test weights and shared files are available via **Baidu Netdisk**:

- **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1FDvIuLA0eJ1rGxUB9B1YSA?pwd=g7yu)
- **Extraction code:** `g7yu`

After downloading, specify the path to the model checkpoint using the `--checkpoint` argument in the inference command below.

## Installation

Clone this repository and install the Python dependencies:

```bash
git clone <THIS_REPOSITORY_URL>
cd Brainseg_github

conda create -n singlebrainseg python=3.10 -y
conda activate singlebrainseg
pip install -r requirements.txt
```

If you use GPU inference, please make sure your CUDA / PyTorch installation is compatible with your system.

## Getting Started via Python

The main testing entry is:

```bash
python backbone/inference_by_csv.py -h
```

### Input CSV

The inference script reads cases from a CSV file. The CSV should contain at least the following columns:

| Column | Description |
| --- | --- |
| `Site` | Site or dataset name. |
| `SubjectID` | Subject identifier. |
| `Session` | Session identifier. |
| `process` | Path to the preprocessed subject folder. |

For each row, the script will look for the input image under:

```text
<process>/step_1_T1w_process_ANTs-2.4.0_synthmorph/T1w2MNI_RigidWarped.nii.gz
```

If this file is not found, it will try:

```text
<process>/step_1_T1w_process_ANTs-2.4.0/T1w2MNI_RigidWarped.nii.gz
```

### Run Inference

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/inference_by_csv.py \
  --args_json path/to/args.json \
  --input_csv path/to/test.csv \
  --output_folder outputs/test_predictions \
  --checkpoint path/to/test_weight.pth \
  --gpu 0 \
  --target_spacing 1 1 1
```

Optional post-processing:

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/inference_by_csv.py \
  --args_json path/to/args.json \
  --input_csv path/to/test.csv \
  --output_folder outputs/test_predictions \
  --checkpoint path/to/test_weight.pth \
  --gpu 0 \
  --target_spacing 1 1 1 \
  --post
```

The output will be saved as:

```text
<output_folder>/<Site>/<SubjectID>/<Session>/<Site>_<SubjectID>_<Session>_SingleBrainSeg.nii.gz
```

If `--post` is enabled, the output will be:

```text
<output_folder>/<Site>/<SubjectID>/<Session>/<Site>_<SubjectID>_<Session>_SingleBrainSeg_post.nii.gz
```

## Repository Structure

```text
Brainseg_github/
├── backbone/             # network, inference, training scaffold, data utilities
├── lookuptable/          # label configuration
├── nnunetv2/             # minimal nnU-Net dependencies used by this project
├── online_iteration/     # Bayesian confidence and online pseudo-label update utilities
├── requirements.txt
└── README.md
```

## Training Code

The complete training pipeline is being cleaned and documented. We will release the full training code after the paper is accepted.

## Citation

If you find this project useful, please consider citing our work. The citation entry will be updated after the paper is accepted.

```bibtex
@misc{singlebrainseg2026,
  title  = {SingleBrainSeg},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## Acknowledgments

This project uses or refers to components from several excellent open-source projects, including nnU-Net, SynthMorph, MONAI, and related medical image analysis toolkits.
