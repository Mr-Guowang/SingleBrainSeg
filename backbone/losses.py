from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .paths import add_project_paths

add_project_paths()

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402
from nnunetv2.utilities.helpers import softmax_helper_dim1  # noqa: E402


def deep_supervision_scales(pool_op_kernel_sizes, enabled: bool = True):
    if not enabled:
        return None
    return list(list(i) for i in 1 / np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0))[:-1]



class ConfidenceWeightedDCAndCELoss(nn.Module):
    """nnU-Net DC+CE with voxel-wise confidence weights for weak labels.

    CE is computed with reduction='none' and normalized by sum(confidence).
    Dice receives confidence as the spatial loss_mask, so TP/FP/FN statistics are
    weighted voxel-by-voxel as well. This is intended for label-map training, not
    region-based BCE targets.
    """

    def __init__(self, batch_dice: bool = False, smooth: float = 1e-5, do_bg: bool = False, ddp: bool = False, confidence_power: float = 1.0):
        super().__init__()
        self.confidence_power = float(confidence_power)
        if self.confidence_power <= 0:
            raise ValueError("confidence_power must be > 0")
        self.dc = MemoryEfficientSoftDiceLoss(
            apply_nonlin=softmax_helper_dim1,
            batch_dice=batch_dice,
            do_bg=do_bg,
            smooth=smooth,
            ddp=ddp,
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor):
        if target.ndim == net_output.ndim - 1:
            target = target[:, None]
        if confidence.ndim == net_output.ndim - 1:
            confidence = confidence[:, None]
        confidence = confidence.to(device=net_output.device, dtype=torch.float32).clamp_min(0)
        if self.confidence_power != 1.0:
            confidence = confidence.pow(self.confidence_power)
        target = target.to(device=net_output.device).long()

        dice_loss = self.dc(net_output, target, loss_mask=confidence)
        ce = F.cross_entropy(net_output, target[:, 0], reduction="none")
        weight = confidence[:, 0]
        ce_loss = (ce * weight).sum() / weight.sum().clamp_min(1e-8)
        return ce_loss + dice_loss


def build_confidence_weighted_loss(label_manager, batch_dice: bool = False, is_ddp: bool = False, confidence_power: float = 1.0) -> nn.Module:
    if label_manager.has_regions:
        raise NotImplementedError("confidence_weight is implemented for label-map CE+Dice, not region BCE targets")
    return ConfidenceWeightedDCAndCELoss(
        batch_dice=batch_dice,
        smooth=1e-5,
        do_bg=False,
        ddp=is_ddp,
        confidence_power=confidence_power,
    )


def build_nnunet_loss(
    label_manager,
    batch_dice: bool = False,
    deep_supervision: bool = False,
    pool_op_kernel_sizes=None,
    is_ddp: bool = False,
    compile_dice: bool = False,
    ignore_label_override=None,
) -> nn.Module:
    ignore_label = label_manager.ignore_label if ignore_label_override is None else ignore_label_override
    if label_manager.has_regions:
        loss = DC_and_BCE_loss(
            {},
            {"batch_dice": batch_dice, "do_bg": True, "smooth": 1e-5, "ddp": is_ddp},
            use_ignore_label=ignore_label is not None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
    else:
        loss = DC_and_CE_loss(
            {"batch_dice": batch_dice, "smooth": 1e-5, "do_bg": False, "ddp": is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    if compile_dice and hasattr(torch, "compile"):
        loss.dc = torch.compile(loss.dc)

    if deep_supervision:
        if pool_op_kernel_sizes is None:
            raise ValueError("pool_op_kernel_sizes is required when deep_supervision=True")
        scales = deep_supervision_scales(pool_op_kernel_sizes, True)
        weights = np.array([1 / (2**i) for i in range(len(scales))])
        weights[-1] = 1e-6 if is_ddp and not compile_dice else 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
    return loss
