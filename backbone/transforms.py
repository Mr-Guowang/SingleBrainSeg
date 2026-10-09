from __future__ import annotations

import csv
import re
from typing import List, Tuple, Union

import numpy as np
import torch

from .paths import add_project_paths

add_project_paths()

from batchgeneratorsv2.helpers.scalar_type import RandomScalar  # noqa: E402
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform  # noqa: E402
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform  # noqa: E402
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform  # noqa: E402
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform  # noqa: E402
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform  # noqa: E402
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform  # noqa: E402
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform  # noqa: E402
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms  # noqa: E402
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.random import RandomTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform  # noqa: E402
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform  # noqa: E402
from nnunetv2.configuration import ANISO_THRESHOLD  # noqa: E402
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size  # noqa: E402


DEFAULT_LEFT_RIGHT_LABEL_PAIRS = [
    (6, 21),
    (7, 22),
    (8, 23),
    (9, 24),
    (10, 25),
    (11, 26),
    (12, 27),
    (13, 28),
    (14, 29),
    (15, 30),
    (16, 31),
    (17, 32),
    (18, 33),
    (19, 34),
    (20, 35),
]


def _normalize_lr_structure_name(name: str):
    text = str(name).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    left_prefixes = ("left ", "lh ", "lh")
    right_prefixes = ("right ", "rh ", "rh")
    for prefix in left_prefixes:
        if text.startswith(prefix):
            return "left", text[len(prefix):].strip()
    for prefix in right_prefixes:
        if text.startswith(prefix):
            return "right", text[len(prefix):].strip()
    return None, text


def infer_left_right_pairs_from_csv(csv_path: str):
    if csv_path is None or str(csv_path).strip() == "":
        return None
    sides = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "Structure" not in reader.fieldnames or "GGBond" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain Structure and GGBond columns")
        for row in reader:
            side, base = _normalize_lr_structure_name(row.get("Structure", ""))
            if side not in ("left", "right") or base == "":
                continue
            label_text = str(row.get("GGBond", "")).strip()
            if label_text == "":
                continue
            try:
                label = int(float(label_text))
            except ValueError:
                continue
            sides.setdefault(base, {})[side] = label
    pairs = []
    for base in sorted(sides):
        item = sides[base]
        if "left" in item and "right" in item:
            pairs.append((int(item["left"]), int(item["right"])))
    if not pairs:
        raise RuntimeError(f"No left/right label pairs could be inferred from {csv_path}")
    return pairs


class FirstNChannelsTransform(BasicTransform):
    """Apply an intensity transform only to selected image channels.

    Spatial transforms see the full image tensor, so confidence stays aligned.
    This wrapper only clones the selected channels, not the full image, which keeps
    it close to the speed of the patched nnU-Net first_n_channels transforms.
    """

    def __init__(self, transform: BasicTransform, channels=(0,)):
        super().__init__()
        self.transform = transform
        self.channels = tuple(int(c) for c in channels)

    def apply(self, data_dict, **params):
        image = data_dict.get("image")
        if image is None:
            return data_dict
        channels = [c for c in self.channels if 0 <= c < image.shape[0]]
        if not channels:
            return data_dict
        if len(channels) == image.shape[0] and channels == list(range(image.shape[0])):
            data_dict["image"] = self.transform(image=image)["image"]
            return data_dict
        transformed = self.transform(image=image[channels].clone())["image"]
        image[channels] = transformed
        data_dict["image"] = image
        return data_dict


class LeftRightMirrorLabelSwapTransform(BasicTransform):
    def __init__(self, left_right_pairs=None, axis: int = 0, p: float = 0.5):
        super().__init__()
        self.left_right_pairs = DEFAULT_LEFT_RIGHT_LABEL_PAIRS if left_right_pairs is None else list(left_right_pairs)
        self.axis = axis
        self.p = p

    def get_parameters(self, **data_dict) -> dict:
        return {"do_mirror": bool(torch.rand(1) < self.p)}

    def _flip(self, x: torch.Tensor, **params) -> torch.Tensor:
        if not params["do_mirror"]:
            return x
        return torch.flip(x, dims=[self.axis + 1])

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return self._flip(img, **params)

    def _apply_to_regr_target(self, regression_target, **params) -> torch.Tensor:
        return self._flip(regression_target, **params)

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> torch.Tensor:
        out = self._flip(segmentation, **params)
        if not params["do_mirror"]:
            return out
        out = out.clone()
        seg_old = out[0].clone()
        seg_new = out[0].clone()
        for left_label, right_label in self.left_right_pairs:
            seg_new[seg_old == left_label] = right_label
            seg_new[seg_old == right_label] = left_label
        out[0] = seg_new
        return out

    def _apply_to_bbox(self, bbox, **params):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **params):
        raise NotImplementedError


def configure_rotation_and_patch_size(patch_size):
    patch_size = np.array(patch_size).astype(int)
    dim = len(patch_size)
    if dim == 2:
        do_dummy_2d_data_aug = False
        if max(patch_size) / min(patch_size) > 1.5:
            rotation_for_da = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
        else:
            rotation_for_da = (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi)
        mirror_axes = (0,)
    elif dim == 3:
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
        if do_dummy_2d_data_aug:
            rotation_for_da = (-180.0 / 360 * 2.0 * np.pi, 180.0 / 360 * 2.0 * np.pi)
        else:
            rotation_for_da = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
        mirror_axes = (0,)
    else:
        raise RuntimeError(f"Only 2D/3D patch sizes are supported, got {patch_size}")

    initial_patch_size = get_patch_size(
        patch_size[-dim:], rotation_for_da, rotation_for_da, rotation_for_da, (0.85, 1.25)
    )
    if do_dummy_2d_data_aug:
        initial_patch_size[0] = patch_size[0]
    return rotation_for_da, do_dummy_2d_data_aug, initial_patch_size.astype(int), mirror_axes


def build_training_transforms(
    patch_size: Union[np.ndarray, Tuple[int]],
    rotation_for_DA: RandomScalar,
    deep_supervision_scales: Union[List, Tuple, None],
    mirror_axes: Tuple[int, ...],
    do_dummy_2d_data_aug: bool,
    use_mask_for_norm: List[bool] = None,
    regions: List[Union[List[int], Tuple[int, ...], int]] = None,
    ignore_label: int = None,
    intensity_channels: Tuple[int, ...] = (0,),
    left_right_pairs=None,
) -> BasicTransform:
    transforms = []
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size
        ignore_axes = None
    transforms.append(
        SpatialTransform(
            patch_size_spatial,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=rotation_for_DA,
            p_scaling=0.2,
            scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
        )
    )
    if do_dummy_2d_data_aug:
        transforms.append(Convert2DTo3DTransform())

    transforms.append(RandomTransform(FirstNChannelsTransform(GaussianNoiseTransform((0, 0.1), p_per_channel=1, synchronize_channels=True), channels=intensity_channels), 0.1))
    transforms.append(RandomTransform(FirstNChannelsTransform(GaussianBlurTransform((0.5, 1.0), False, False, 0.5, benchmark=True), channels=intensity_channels), 0.2))
    transforms.append(RandomTransform(FirstNChannelsTransform(MultiplicativeBrightnessTransform(BGContrast((0.75, 1.25)), False, 1), channels=intensity_channels), 0.15))
    transforms.append(RandomTransform(FirstNChannelsTransform(ContrastTransform(BGContrast((0.75, 1.25)), True, False, 1), channels=intensity_channels), 0.15))
    transforms.append(RandomTransform(FirstNChannelsTransform(SimulateLowResolutionTransform((0.5, 1), False, True, ignore_axes, None, 0.5), channels=intensity_channels), 0.25))
    transforms.append(RandomTransform(FirstNChannelsTransform(GammaTransform(BGContrast((0.7, 1.5)), 1, False, 1, 1), channels=intensity_channels), 0.1))
    transforms.append(RandomTransform(FirstNChannelsTransform(GammaTransform(BGContrast((0.7, 1.5)), 0, False, 1, 1), channels=intensity_channels), 0.3))
    if mirror_axes is not None and 0 in mirror_axes:
        transforms.append(LeftRightMirrorLabelSwapTransform(left_right_pairs=left_right_pairs, axis=0, p=0.5))
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(
            MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            )
        )
    transforms.append(RemoveLabelTansform(-1, 0))
    if regions is not None:
        transforms.append(
            ConvertSegmentationToRegionsTransform(
                regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                channel_in_seg=0,
            )
        )
    if deep_supervision_scales is not None:
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    return ComposeTransforms(transforms)
