from __future__ import annotations

import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .paths import add_project_paths

add_project_paths()

from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd  # noqa: E402
from batchgenerators.dataloading.data_loader import DataLoader  # noqa: E402


class SynWeakDataLoader(DataLoader):
    """
    nnU-Net patch sampler with one extra optional confidence map.

    The bbox sampling, padding value conventions, foreground oversampling,
    and transform application mirror nnUNetDataLoader. Confidence is cropped
    with the same bbox and returned in the batch, but is intentionally not
    passed into the nnU-Net loss.
    """

    def __init__(
        self,
        data,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
        label_manager,
        oversample_foreground_percent: float = 0.33,
        sampling_probabilities=None,
        pad_sides=None,
        probabilistic_oversampling: bool = False,
        transforms=None,
    ):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        self.indices = data.identifiers
        self.patch_size_was_2d = len(patch_size) == 2
        if self.patch_size_was_2d:
            final_patch_size = (1, *patch_size)
            patch_size = (1, *patch_size)

        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = np.array(final_patch_size).astype(int)
        self.patch_size = np.array(patch_size).astype(int)
        self.need_to_pad = (self.patch_size - self.final_patch_size).astype(int)
        if pad_sides is not None:
            if self.patch_size_was_2d:
                pad_sides = (0, *pad_sides)
            for d in range(len(self.need_to_pad)):
                self.need_to_pad[d] += pad_sides[d]

        self.data_shape, self.seg_shape, self.conf_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = tuple([-1] + label_manager.all_labels)
        self.has_ignore = label_manager.has_ignore_label
        self.get_do_oversample = (
            self._oversample_last_XX_percent if not probabilistic_oversampling else self._probabilistic_oversampling
        )
        self.transforms = transforms

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        data, seg, confidence, properties = self._data.load_case(self._data.identifiers[0])
        data_shape = (self.batch_size, data.shape[0], *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        conf_shape = None if confidence is None else (self.batch_size, confidence.shape[0], *self.patch_size)
        return data_shape, seg_shape, conf_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None]):
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)
        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            selected_class = None
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    warnings.warn("Warning! No annotated pixels in image!")
                    selected_class = None
            elif force_fg:
                eligible = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible]
                if any(tmp) and len(eligible) > 1:
                    eligible.pop(np.where(tmp)[0][0])
                if len(eligible) > 0:
                    selected_class = eligible[np.random.choice(len(eligible))]
            if selected_class is not None:
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        conf_all = None if self.conf_shape is None else np.zeros(self.conf_shape, dtype=np.float32)

        for j, identifier in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, confidence, properties = self._data.load_case(identifier)
            bbox_lbs, bbox_ubs = self.get_bbox(data.shape[1:], force_fg, properties["class_locations"])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]
            data_all[j] = crop_and_pad_nd(data, bbox, 0)
            seg_all[j] = crop_and_pad_nd(seg, bbox, -1)
            if conf_all is not None:
                conf_all[j] = crop_and_pad_nd(confidence, bbox, 0)

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            if conf_all is not None:
                conf_all = conf_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_t = torch.from_numpy(data_all).float()
                    seg_t = torch.from_numpy(seg_all).to(torch.int16)
                    conf_t = None if conf_all is None else torch.from_numpy(conf_all).float()
                    images, segs, confs = [], [], []
                    for b in range(self.batch_size):
                        image = data_t[b] if conf_t is None else torch.cat((data_t[b], conf_t[b]), dim=0)
                        tmp = self.transforms(**{"image": image, "segmentation": seg_t[b]})
                        transformed_image = tmp["image"]
                        if conf_t is None:
                            images.append(transformed_image)
                        else:
                            images.append(transformed_image[: data_t.shape[1]])
                            confs.append(transformed_image[data_t.shape[1] : data_t.shape[1] + conf_t.shape[1]].float())
                        segs.append(tmp["segmentation"])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    if conf_t is not None:
                        conf_all = torch.stack(confs).clamp_(0, 1)
        batch = {"data": data_all, "target": seg_all, "keys": selected_keys}
        if conf_all is not None:
            if isinstance(conf_all, list):
                batch["confidence"] = conf_all
            else:
                batch["confidence"] = conf_all if torch.is_tensor(conf_all) else torch.from_numpy(conf_all).float()
        return batch


def infinite_loader(loader: SynWeakDataLoader):
    while True:
        yield loader.generate_train_batch()
