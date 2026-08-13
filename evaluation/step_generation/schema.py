"""Helpers for validating the public Time-Aware MRI benchmark JSON schema."""
from typing import Any, Dict, List

REQUIRED_IMAGE_FIELDS = ("path", "timepoint", "sequence", "view", "filename")


def validate_sample_schema(sample: Dict[str, Any]) -> None:
    """Raise ValueError when required image metadata is missing."""
    images = sample.get("images")
    if not isinstance(images, list) or not images:
        return
    for i, image in enumerate(images):
        missing = [
            k for k in REQUIRED_IMAGE_FIELDS
            if k not in image or image[k] in (None, "")
        ]
        if missing:
            qid = sample.get("qa_id") or sample.get("id") or "unknown"
            raise ValueError(
                f"{qid}: images[{i}] missing required fields: {', '.join(missing)}"
            )


def group_images_by_timepoint(
    sample: Dict[str, Any]
) -> Dict[Any, List[Dict[str, Any]]]:
    validate_sample_schema(sample)
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    for image in sample.get("images", []):
        grouped.setdefault(image["timepoint"], []).append(image)
    return grouped
