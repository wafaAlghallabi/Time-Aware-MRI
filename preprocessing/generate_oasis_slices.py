#!/usr/bin/env python3
"""
OASIS-2 preprocessing for the Time-Aware Multi-View MRI benchmark.

This script:
1. Finds OASIS-2 longitudinal sessions (e.g., OAS2_0001_MR1).
2. Loads the selected raw MPRAGE Analyze image (.img/.hdr).
3. Reorients the volume to canonical RAS orientation.
4. Robustly normalizes the full 3D volume.
5. Estimates the brain center.
6. Extracts axial, coronal, and sagittal thick-slab views.
7. Saves centered 256x256 grayscale PNGs.

Notes
-----
- By default, only mpr-1 is used from each visit. This avoids silently
  overwriting repeated MPRAGE acquisitions within the same visit.
- If multiple repeated MPRAGE acquisitions are to be combined, they should
  first be registered to a common within-visit space before averaging.
- Source MRI data are not redistributed by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage


def robust_normalize_volume(
    volume: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Robustly normalize a 3D MRI volume to [0, 255]."""
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    foreground = volume[np.isfinite(volume) & (volume > 0)]
    if foreground.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)

    lo = np.percentile(foreground, lower_percentile)
    hi = np.percentile(foreground, upper_percentile)

    if hi <= lo:
        return np.zeros_like(volume, dtype=np.uint8)

    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    volume = np.clip(volume, 0.0, 1.0)

    # Keep the original zero background black.
    volume[np.asarray(volume) <= 0] = 0

    return np.round(volume * 255.0).astype(np.uint8)


def get_brain_center(
    volume: np.ndarray,
    threshold_percentile: float = 50.0,
) -> Tuple[int, int, int]:
    """
    Estimate the brain center from a normalized 3D T1-weighted volume.

    The returned coordinates follow canonical RAS array order:
        axis 0 -> left/right   (sagittal slicing)
        axis 1 -> posterior/anterior (coronal slicing)
        axis 2 -> inferior/superior  (axial slicing)
    """
    foreground = volume[volume > 0]

    if foreground.size == 0:
        return tuple(int(s // 2) for s in volume.shape)

    threshold = np.percentile(foreground, threshold_percentile)
    mask = volume > threshold

    if not np.any(mask):
        return tuple(int(s // 2) for s in volume.shape)

    center = ndimage.center_of_mass(mask)
    return tuple(int(round(c)) for c in center)


def extract_thick_slab(
    volume: np.ndarray,
    slice_idx: int,
    axis: int,
    slab_radius: int = 2,
) -> np.ndarray:
    """
    Extract a mean-projected thick slab.

    slab_radius=2 means 5 slices total:
        center-2, center-1, center, center+1, center+2
    """
    if axis < 0 or axis >= volume.ndim:
        raise ValueError(f"Invalid axis {axis} for volume with shape {volume.shape}")

    slice_idx = int(np.clip(slice_idx, 0, volume.shape[axis] - 1))
    start = max(slice_idx - slab_radius, 0)
    end = min(slice_idx + slab_radius + 1, volume.shape[axis])

    slicer = [slice(None)] * volume.ndim
    slicer[axis] = slice(start, end)

    slab = volume[tuple(slicer)]
    result = np.mean(slab, axis=axis)

    if result.ndim != 2:
        result = np.squeeze(result)

    if result.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {result.shape}")

    return result


def resize_and_pad(img: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Resize a 2D image while preserving aspect ratio and pad with black.

    No brain-specific auto-cropping is applied. This avoids changing the
    apparent anatomy scale independently between longitudinal visits.
    """
    img = np.asarray(img)

    if img.ndim != 2:
        img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}")

    img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    scale = min(target_size / h, target_size / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    pil_img = Image.fromarray(img, mode="L")
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.asarray(pil_img, dtype=np.uint8)

    output = np.zeros((target_size, target_size), dtype=np.uint8)

    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    output[y0:y0 + new_h, x0:x0 + new_w] = resized

    return output


def orient_for_display(img: np.ndarray, view: str) -> np.ndarray:
    """
    Apply a consistent display orientation after canonical RAS conversion.

    This does not affect which anatomical plane is extracted; it only makes
    the saved PNG orientation consistent across cases.
    """
    if view == "axial":
        return np.rot90(img)
    if view == "coronal":
        return np.rot90(img)
    if view == "sagittal":
        return np.rot90(img)
    raise ValueError(f"Unknown view: {view}")


def save_png(img: np.ndarray, output_path: Path, target_size: int = 256) -> None:
    """Resize/pad and save a 2D grayscale PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = resize_and_pad(img, target_size=target_size)
    Image.fromarray(img, mode="L").save(output_path, format="PNG", compress_level=1)


def select_mpr_file(raw_dir: Path, acquisition: str) -> Path | None:
    """
    Select one MPRAGE Analyze image from an OASIS-2 visit.

    acquisition examples:
        "mpr-1"
        "mpr-2"
        "mpr-3"
    """
    preferred = raw_dir / f"{acquisition}.img"
    if preferred.exists():
        return preferred

    available = sorted(raw_dir.glob("mpr-*.img"))
    if not available:
        return None

    # Fallback to the first available scan if the requested acquisition
    # is missing, while keeping the behavior explicit in the console output.
    return available[0]


def load_canonical_volume(img_file: Path) -> np.ndarray:
    """Load an Analyze/NIfTI image and reorient it to canonical RAS."""
    img = nib.load(str(img_file))
    img = nib.as_closest_canonical(img)

    volume = np.squeeze(img.get_fdata(dtype=np.float32))

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume after squeeze, got shape {volume.shape}"
        )

    return volume


def generate_views(
    volume: np.ndarray,
    slab_radius: int = 2,
) -> Dict[str, np.ndarray]:
    """Generate axial, coronal, and sagittal views from canonical RAS volume."""
    normalized = robust_normalize_volume(volume)
    x_center, y_center, z_center = get_brain_center(normalized)

    views = {
        # Canonical RAS array axes:
        # axis 0 = sagittal, axis 1 = coronal, axis 2 = axial
        "axial": extract_thick_slab(
            normalized, z_center, axis=2, slab_radius=slab_radius
        ),
        "coronal": extract_thick_slab(
            normalized, y_center, axis=1, slab_radius=slab_radius
        ),
        "sagittal": extract_thick_slab(
            normalized, x_center, axis=0, slab_radius=slab_radius
        ),
    }

    return {name: orient_for_display(img, name) for name, img in views.items()}


def parse_session_name(session_name: str) -> Tuple[str, str, int]:
    """
    Parse an OASIS-2 session name.

    Example:
        OAS2_0001_MR1 -> ("OAS2_0001", "MR1", 0)
    """
    parts = session_name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid OASIS-2 session name: {session_name}")

    patient_id = f"{parts[0]}_{parts[1]}"
    timepoint = parts[2]

    if not timepoint.startswith("MR"):
        raise ValueError(f"Invalid OASIS-2 timepoint: {timepoint}")

    tp_index = int(timepoint.replace("MR", "")) - 1
    return patient_id, timepoint, tp_index


def process_oasis_dataset(
    base_path: Path,
    out_path: Path,
    acquisition: str = "mpr-1",
    slab_radius: int = 2,
    target_size: int = 256,
) -> None:
    """Generate longitudinal multi-view PNGs from OASIS-2 raw sessions."""
    base_path = Path(base_path)
    out_path = Path(out_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    sessions = sorted(
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith("OAS2_")
    )

    stats = {
        "total_sessions": len(sessions),
        "processed": 0,
        "failed": 0,
        "total_images": 0,
    }

    print(f"Found {len(sessions)} OASIS-2 sessions.")

    for session_dir in sessions:
        try:
            patient_id, timepoint, tp_index = parse_session_name(session_dir.name)
            raw_dir = session_dir / "RAW"

            if not raw_dir.exists():
                raise FileNotFoundError(f"RAW directory not found: {raw_dir}")

            selected = select_mpr_file(raw_dir, acquisition)
            if selected is None:
                raise FileNotFoundError(f"No mpr-*.img files found in {raw_dir}")

            if selected.name != f"{acquisition}.img":
                print(
                    f"[WARN] {session_dir.name}: {acquisition}.img not found; "
                    f"using {selected.name}."
                )

            volume = load_canonical_volume(selected)
            views = generate_views(volume, slab_radius=slab_radius)

            patient_out = out_path / patient_id
            patient_out.mkdir(parents=True, exist_ok=True)

            for view_name, img in views.items():
                filename = f"{patient_id}_{tp_index}_t1_{view_name}.png"
                save_png(
                    img,
                    patient_out / filename,
                    target_size=target_size,
                )
                stats["total_images"] += 1

            stats["processed"] += 1
            print(
                f"[OK] {session_dir.name} | {selected.name} | "
                f"shape={volume.shape} | saved=3"
            )

        except Exception as exc:
            stats["failed"] += 1
            print(f"[ERROR] {session_dir.name}: {exc}")

    print("\nProcessing complete")
    print(f"  Total sessions : {stats['total_sessions']}")
    print(f"  Processed      : {stats['processed']}")
    print(f"  Failed         : {stats['failed']}")
    print(f"  Total images   : {stats['total_images']}")
    print(f"  Output         : {out_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical multi-view PNGs from OASIS-2 raw MRI."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the OAS2_RAW directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where generated PNGs will be saved.",
    )
    parser.add_argument(
        "--acquisition",
        default="mpr-1",
        help="Within-visit MPRAGE acquisition to use (default: mpr-1).",
    )
    parser.add_argument(
        "--slab-radius",
        type=int,
        default=2,
        help="Number of neighboring slices on each side of the center (default: 2 = 5-slice slab).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Output PNG size in pixels (default: 256).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.slab_radius < 0:
        raise ValueError("--slab-radius must be >= 0")

    if args.target_size <= 0:
        raise ValueError("--target-size must be > 0")

    process_oasis_dataset(
        base_path=args.input,
        out_path=args.output,
        acquisition=args.acquisition,
        slab_radius=args.slab_radius,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
