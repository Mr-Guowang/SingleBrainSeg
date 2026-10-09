from __future__ import annotations

from .paths import add_project_paths

add_project_paths()

from nnunetv2.utilities.label_handling.label_handling import LabelManager  # noqa: E402


def build_label_manager(dataset_json: dict) -> LabelManager:
    return LabelManager(
        dataset_json["labels"],
        dataset_json.get("regions_class_order"),
        force_use_labels=dataset_json.get("force_use_labels", False),
    )
