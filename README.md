# SingleBrainSeg

SingleBrainSeg is a brain MRI segmentation framework designed for robust anatomical segmentation with pseudo-label refinement and structural subspace guidance.

This repository is under active maintenance. We will continue to update the code, documentation, pretrained weights, and examples.

## Latest Updates

| Date | Update |
| --- | --- |
| 2026-10-11 | Update the training code. |
| 2026-10-10 | Update the testing code and test weights. Now you can try on the test data / your own data. |
| 2026-10-09 | Initial GitHub release with Python code. |

## Notice

This release provides the testing/inference code, test weights, and training code.

We are still polishing the repository for public use. The code, documentation, pretrained weights, and examples will be continuously updated.

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
  --modelverson paper \
  --taskcode example_training \
  --dataset_json path/to/dataset.json \
  --num_classes 36 \
  --synth_pretrained /absolute/path/to/initial_pretrained_checkpoint.pth \
  --simulatedir /path/to/supervised_processed_data \
  --confidence_iter_input_csv /path/to/training_subjects.csv \
  --confidence_iter_output_root /path/to/online_iteration_output \
  --confidence_iter_tissue_csv /path/to/subspace_table.csv \
  --confidence_iter_prior_dir /path/to/initial_confidence_or_prior_dir \
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

Key training arguments:

| Argument | Description |
| --- | --- |
| `--synth_pretrained` | Initial checkpoint used to initialize the model. |
| `--simulatedir` | Fully supervised processed data used in warm-up and joint training. |
| `--confidence_iter_input_csv` | Subject table used during online pseudo-label refresh. |
| `--confidence_iter_output_root` | Directory where online pseudo-labels, confidence maps, and iterative synth labels are saved. |
| `--confidence_iter_tissue_csv` | Tissue grouping table for Bayesian confidence estimation. |
| `--confidence_iter_prior_dir` | Initial prior/confidence directory used to locate prepared subspace resources. |
| `--left_right_pairs_csv` | Look-up table for left/right label pairs. |
| `--warmup_epochs` | Number of supervised warm-up epochs. |
| `--num_epochs` | Total number of epochs. |
| `--confidence_iter_interval` | Bayesian pseudo-label refresh interval. |

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
