
# SingleBrainSeg

### One Annotation. Your Protocol. Your Brain Segmenter.

**SingleBrainSeg** is a label-efficient framework for building **customized 3D brain MRI segmentation models from a single annotated subject**.

By leveraging unlabeled images, SingleBrainSeg enables adaptation to different anatomical labeling protocols without requiring extensive manual annotations.

Our goal is to make protocol-specific brain segmentation **more accessible, flexible, and practical** for the neuroimaging community.

## 🧠 Supported Labeling Protocols

The current release supports inference under three anatomical labeling protocols:

- **FreeSurfer** — Adult whole-brain segmentation
- **MALC12** — Multi-atlas brain labeling protocol
- **M-CRIB** — Neonatal brain segmentation

## 🚀 Ongoing Development

**SingleBrainSeg is an actively evolving research project.**

We are continuously exploring additional anatomical labeling protocols, MRI modalities, and pretrained models.

Through these efforts, we hope to expand the applicability of SingleBrainSeg and contribute reusable resources to the broader neuroimaging community.

**Code Availability:** The current repository provides inference code and pretrained model weights. The complete training code will be released upon acceptance of our manuscript.

## 🤝 Collaboration & Community

**We warmly welcome collaborations and contributions!**

If you are working with a customized anatomical labeling protocol, a different MRI modality, or a dataset that could benefit from SingleBrainSeg, we would be delighted to explore potential collaborations.

Feel free to open an issue or reach out to us.

**Let's work together to make brain segmentation more accessible across diverse research applications!**


## Latest Updates

| Date | Update |
| --- | --- |
| 2026-10-11 | Update the training code. |
| 2026-10-10 | Update the testing code and test weights. Now you can try on the test data / your own data. |
| 2026-10-09 | Initial GitHub release with Python code. |

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

## Notice

This release provides the testing/inference code, test weights, and training code.

We are still polishing the repository for public use. The code, documentation, pretrained weights, and examples will be continuously updated.

## Download Test Weights

The test weights and shared files are available via **Baidu Netdisk**:

- **Download link:** [Baidu Netdisk](https://pan.baidu.com/s/1FDvIuLA0eJ1rGxUB9B1YSA?pwd=g7yu)
- **Extraction code:** `g7yu`

After downloading, specify the absolute path to the model checkpoint using the `--checkpoint` argument in the inference command below.

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

## Repository Structure

```text
Brainseg_github/
├── backbone/             # network, inference, training scaffold, data utilities
├── nnunetv2/             # minimal nnU-Net dependencies used by this project
├── online_iteration/     # Bayesian confidence and online pseudo-label update utilities
├── requirements.txt
└── README.md
```

## Training Code

The training entry is:

```bash
python backbone/training_paper.py -h
```

The default training schedule is:

- 100 epochs of supervised warm-up.
- 300 epochs of joint pseudo-label learning, supervised learning, synthetic teacher regularization, and feature distillation.
- Bayesian pseudo-label refresh every 10 epochs by default.

Example command:

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/training_paper.py \
  --modelverson test \
  --taskcode example_training \
  --dataset_json path/to/dataset.json \
  --num_classes 36 \
  --left_right_pairs_csv /path/to/Brain_lookuptable.csv \
  --confidence_iter_gpu 0 \
  --confidence_iter_target_spacing 1 1 1 \
  --num_epochs 400 \
  --warmup_epochs 100 \
  --confidence_iter_interval 10 \
  --save_every 10 \
  --deep_supervision
```

For infant or high-resolution data, use:

```bash
--confidence_iter_target_spacing 0.5 0.5 0.5
```
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
