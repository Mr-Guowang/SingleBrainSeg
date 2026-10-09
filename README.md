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

After downloading, specify the absolute path to the model checkpoint using the `--checkpoint` argument in the inference command below.

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
python backbone/inference.py -h
```

### Preprocessing Requirement

Before running SingleBrainSeg, the input T1-weighted MRI should be preprocessed as follows:

- N4 bias field correction using ANTs.
- Skull stripping using SynthStrip.
- Rigid registration to the template using ANTs.

The inference input should be the preprocessed image in the template-aligned space.

### Run Inference

Minimal example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/inference.py \
  --dataset_json path/to/dataset.json \
  --lookuptable_csv path/to/Brain_lookuptable.csv \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --image path/to/preprocessed_T1w.nii.gz \
  --out outputs/segmentation.nii.gz \
  --gpu 0 \
  --target_spacing 1 1 1 \
  --post
```

For infant or high-resolution data, use the corresponding target spacing, for example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/inference.py \
  --dataset_json path/to/dataset.json \
  --lookuptable_csv path/to/Brain_lookuptable.csv \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --image path/to/preprocessed_T1w.nii.gz \
  --out outputs/segmentation.nii.gz \
  --gpu 0 \
  --target_spacing 0.5 0.5 0.5 \
  --post
```

Key arguments:

| Argument | Description |
| --- | --- |
| `--dataset_json` | Dataset label definition file. |
| `--lookuptable_csv` | Look-up table used for left/right label pairs and post-processing. |
| `--checkpoint` | Absolute path to the model checkpoint. |
| `--image` | Preprocessed input image. |
| `--out` | Output segmentation path. |
| `--target_spacing` | Inference spacing. Use `1 1 1` for standard adult data and `0.5 0.5 0.5` for high-resolution infant-style data. |
| `--post` | Enable connected-component post-processing. |

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
