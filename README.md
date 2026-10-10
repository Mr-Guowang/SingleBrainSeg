
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

We will continue to maintain and improve SingleBrainSeg. If you have any questions, encounter issues, or have suggestions, please feel free to open an issue on GitHub. We welcome your feedback and contributions!

## 🤝 Collaboration & Community

**We warmly welcome collaborations and contributions!**

If you are working with a customized anatomical labeling protocol, a different MRI modality, or a dataset that could benefit from SingleBrainSeg, we would be delighted to explore potential collaborations.

Feel free to open an issue or reach out to us.

**Let's work together to make brain segmentation more accessible across diverse research applications!**



## 📢 Latest Updates

We are continuously improving SingleBrainSeg and expanding its functionality. Stay tuned for future updates!

| Date | Update |
|:---:|---|
| **2026-10-11** |  Released the training code for SingleBrainSeg. |
| **2026-10-10** |  Released the inference pipeline and pretrained model weights. You can now test SingleBrainSeg on our example data or your own MRI scans. |
| **2026-10-09** |  Initial release of the SingleBrainSeg GitHub repository. |

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Mr-Guowang/SingleBrainSeg.git
cd SingleBrainSeg

conda create -n singlebrainseg python=3.10 -y
conda activate singlebrainseg

pip install -r requirements.txt
```

> [!NOTE]
> For GPU-accelerated inference and training, please ensure that your PyTorch installation is compatible with your CUDA version and GPU drivers.

---

## ⚡ Quick Start

SingleBrainSeg provides a Python-based inference pipeline for **protocol-specific 3D brain MRI segmentation**.

To explore the available inference options:

```bash
python backbone/inference.py -h
```

### 🧩 Step 1. Prepare Your MRI Data

Before running inference, input MRI scans should undergo the following preprocessing steps:

1. **N4 Bias Field Correction** — Correct intensity inhomogeneity using ANTs.
2. **Skull Stripping** — Extract brain tissue using SynthStrip.
3. **Rigid Registration** — Align the brain image to the corresponding template using ANTs.

The input to SingleBrainSeg should be a **preprocessed MRI volume aligned to the appropriate template space**.

The current framework supports adult T1-weighted MRI and neonatal T2-weighted MRI under their corresponding labeling protocols.

### 📦 Step 2. Download Pretrained Model Weights

Pretrained checkpoints and shared resources are available through **Baidu Netdisk** , Currently supports three labeling protocols: Freesurfer, MALC12, and M-CRIB:

- 🔗 **Download:** [SingleBrainSeg Model Weights](https://pan.baidu.com/s/1FDvIuLA0eJ1rGxUB9B1YSA?pwd=g7yu)
- 🔑 **Extraction Code:** `g7yu`

After downloading, specify the absolute path to the corresponding model checkpoint using the `--checkpoint` argument.

> [!IMPORTANT]
> Make sure to select the checkpoint, dataset configuration, and lookup table corresponding to your target labeling protocol.

### 🧠 Step 3. Run Inference

**Example 1: Adult brain MRI (1.0 mm isotropic resolution)**

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

**Example 2: Neonatal brain MRI (0.5 mm isotropic resolution)**

```bash
CUDA_VISIBLE_DEVICES=0 \
python backbone/inference.py \
  --dataset_json path/to/dataset.json \
  --lookuptable_csv path/to/Brain_lookuptable.csv \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --image path/to/preprocessed_T2w.nii.gz \
  --out outputs/segmentation.nii.gz \
  --gpu 0 \
  --target_spacing 0.5 0.5 0.5 \
  --post
```

The resulting segmentation will be saved as a NIfTI file at the specified output path.

For additional inference options, run:

```bash
python backbone/inference.py --help
```


## 🏋️ Training

SingleBrainSeg provides a training pipeline for **customizing brain segmentation models using a single annotated subject and unlabeled MRI data**.

### Training Overview

The default training procedure consists of two stages:

**Stage 1: Supervised Warm-up (100 epochs)**

Establish an initial segmentation model using supervised training with the available annotation and augmented image–label pairs.

**Stage 2: Semi-Supervised Optimization (300 epochs)**

By default, Bayesian pseudo-labels and their confidence estimates are refreshed every **10 epochs**.

### Training Configuration

To view available training parameters:

```bash
python backbone/training_paper.py -h
```

### 🚀 Example Training Command

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

For neonatal or high-resolution MRI data, modify the corresponding spacing parameter:

```bash
--confidence_iter_target_spacing 0.5 0.5 0.5
```

> [!NOTE]
> Please adjust the number of classes, labeling protocol configuration, lookup table, and image spacing according to your target dataset.

---

## 📚 Citation

If you find SingleBrainSeg useful in your research, please consider citing our work.

The official citation will be updated upon publication.

```bibtex
@misc{singlebrainseg2026,
  title  = {SingleBrainSeg},
  author = {Anonymous},
  year   = {2026},
  note   = {Manuscript under review}
}
```

---

## 🙏 Acknowledgments

SingleBrainSeg builds upon and benefits from several outstanding open-source projects and research frameworks, including:

- **nnU-Net** — Self-configuring deep learning for biomedical image segmentation.
- **SynthMorph** — Learning-based deformable image registration.
- **MONAI** — Open-source framework for deep learning in medical imaging.

We sincerely thank the developers and research communities behind these projects for making their work publicly available.

Their contributions have greatly supported the development of SingleBrainSeg.
