from __future__ import annotations

import json
from typing import Tuple

from .dataloader import SynWeakDataLoader, infinite_loader
from .dataset import (
    NiftiPatchDataset,
    PreprocessedPatchDataset,
    discover_preprocessed_cases,
    discover_simulate_cases,
    discover_weak_cases,
    is_preprocessed_dataset,
    load_preprocessed_dataset_metadata,
)
from .label_manager import build_label_manager
from .losses import deep_supervision_scales
from .transforms import build_training_transforms, configure_rotation_and_patch_size, infer_left_right_pairs_from_csv

from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter  # noqa: E402
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA  # noqa: E402


DEFAULT_POOL_OP_KERNEL_SIZES = [
    [1, 1, 1],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]


def load_dataset_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_nnunet_train_transform(
    patch_size,
    pool_op_kernel_sizes=DEFAULT_POOL_OP_KERNEL_SIZES,
    deep_supervision: bool = False,
    label_manager=None,
    intensity_channels=(0,),
    load_weak_confidence: bool = True,
    left_right_pairs=None,
    left_right_pairs_csv: str | None = None,
):
    rotation, do_dummy_2d, initial_patch_size, mirror_axes = configure_rotation_and_patch_size(patch_size)
    if left_right_pairs is None and left_right_pairs_csv is not None:
        left_right_pairs = infer_left_right_pairs_from_csv(left_right_pairs_csv)
    ds_scales = deep_supervision_scales(pool_op_kernel_sizes, deep_supervision)
    regions = label_manager.foreground_regions if label_manager is not None and label_manager.has_regions else None
    ignore_label = label_manager.ignore_label if label_manager is not None else None
    transforms = build_training_transforms(
        patch_size,
        rotation,
        ds_scales,
        mirror_axes,
        do_dummy_2d,
        use_mask_for_norm=[False],
        regions=regions,
        ignore_label=ignore_label,
        intensity_channels=tuple(int(i) for i in intensity_channels),
        left_right_pairs=left_right_pairs,
    )
    return transforms, initial_patch_size


def get_data_loader(
    weak_dir: str,
    simulate_dir: str,
    dataset_json: str,
    patch_size: Tuple[int, int, int],
    batch_size: int,
    oversample_foreground_percent: float = 0.33,
    deep_supervision: bool = False,
    pool_op_kernel_sizes=DEFAULT_POOL_OP_KERNEL_SIZES,
    num_processes_da: int | None = None,
    pin_memory: bool = False,
    intensity_channels=(0,),
    load_weak_confidence: bool = True,
    left_right_pairs=None,
    left_right_pairs_csv: str | None = None,
):
    dataset_info = load_dataset_json(dataset_json)
    label_manager = build_label_manager(dataset_info)
    transforms, initial_patch_size = build_nnunet_train_transform(
        patch_size,
        pool_op_kernel_sizes,
        deep_supervision,
        label_manager,
        intensity_channels=intensity_channels,
        left_right_pairs=left_right_pairs,
        left_right_pairs_csv=left_right_pairs_csv,
    )

    weak_is_fast = is_preprocessed_dataset(weak_dir)
    simulate_is_fast = is_preprocessed_dataset(simulate_dir)
    if weak_is_fast:
        weak_meta = load_preprocessed_dataset_metadata(weak_dir)
        weak_dataset = PreprocessedPatchDataset(
            discover_preprocessed_cases(weak_dir),
            labels=label_manager.all_labels,
            ignore_label=label_manager.ignore_label,
            data_channels=weak_meta.get("data_channels"),
            confidence_channel=weak_meta.get("confidence_channel", 1) if load_weak_confidence else None,
        )
    else:
        weak_dataset = NiftiPatchDataset(
            discover_weak_cases(weak_dir),
            labels=label_manager.all_labels,
            ignore_label=label_manager.ignore_label,
            normalize=False,
        )
        if not load_weak_confidence:
            weak_dataset.cases = {k: type(v)(v.identifier, v.image, v.seg, None) for k, v in weak_dataset.cases.items()}


    if simulate_is_fast:
        simulate_meta = load_preprocessed_dataset_metadata(simulate_dir)
        simulate_dataset = PreprocessedPatchDataset(
            discover_preprocessed_cases(simulate_dir),
            labels=label_manager.all_labels,
            ignore_label=label_manager.ignore_label,
            data_channels=simulate_meta.get("data_channels"),
            confidence_channel=simulate_meta.get("confidence_channel"),
        )
    else:
        simulate_dataset = NiftiPatchDataset(
            discover_simulate_cases(simulate_dir),
            labels=label_manager.all_labels,
            ignore_label=label_manager.ignore_label,
            normalize=True,
        )

    weak_loader = SynWeakDataLoader(
        weak_dataset,
        batch_size,
        initial_patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=oversample_foreground_percent,
        transforms=transforms,
    )
    simulate_loader = SynWeakDataLoader(
        simulate_dataset,
        batch_size,
        initial_patch_size,
        patch_size,
        label_manager,
        oversample_foreground_percent=oversample_foreground_percent,
        transforms=transforms,
    )
    if num_processes_da is None:
        num_processes_da = get_allowed_n_proc_DA()
    num_processes_da = int(num_processes_da)
    print(
        f"[dataloader] weak_fast={weak_is_fast}, simulate_fast={simulate_is_fast}, load_weak_confidence={load_weak_confidence}, "
        f"initial_patch_size={tuple(int(i) for i in initial_patch_size)}, final_patch_size={tuple(int(i) for i in patch_size)}, "
        f"num_processes_da={num_processes_da}, pin_memory={pin_memory}"
    )
    if num_processes_da <= 0:
        return infinite_loader(weak_loader), infinite_loader(simulate_loader)

    weak_mt = NonDetMultiThreadedAugmenter(
        data_loader=weak_loader,
        transform=None,
        num_processes=num_processes_da,
        num_cached=max(6, num_processes_da // 2),
        seeds=None,
        pin_memory=pin_memory,
        wait_time=0.002,
    )
    simulate_processes = max(1, num_processes_da // 2)
    simulate_mt = NonDetMultiThreadedAugmenter(
        data_loader=simulate_loader,
        transform=None,
        num_processes=simulate_processes,
        num_cached=max(3, simulate_processes // 2),
        seeds=None,
        pin_memory=pin_memory,
        wait_time=0.002,
    )
    return weak_mt, simulate_mt
