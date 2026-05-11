#!/usr/bin/env python3
"""Export a colored PLY point cloud directly from images with local VGGT."""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from utils.sh_utils import RGB2SH

REPO_ROOT = Path(__file__).resolve().parent
VGGT_ROOT = REPO_ROOT / "submodules" / "vggt"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

HF_HOME = "/root/autodl-tmp/huggingface"
HF_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
DEFAULT_MODEL_DIR = "/root/autodl-tmp/huggingface/hub/models--facebook--VGGT-1B"

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def configure_hf_cache() -> None:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)


def collect_images(image_dir: str) -> List[str]:
    image_paths: List[str] = []
    for pattern in IMAGE_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern)))
        image_paths.extend(glob.glob(os.path.join(image_dir, pattern.upper())))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return image_paths


def squeeze_scene_tensor(value: torch.Tensor, expected_images: int, name: str) -> torch.Tensor:
    """Normalize VGGT output tensors to shapes starting with image dimension S."""
    if value.dim() >= 1 and value.shape[0] == 1:
        value = value[0]
    if value.dim() >= 1 and value.shape[0] != expected_images:
        raise RuntimeError(f"Unexpected {name} shape {tuple(value.shape)} for {expected_images} images")
    return value


def tensor_images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Convert [S,3,H,W] float images in [0,1] to [S,H,W,3] uint8."""
    arr = images.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def load_model(model_name: str, model_dir: str, device: str):
    from vggt.models.vggt import VGGT

    configure_hf_cache()

    local_candidates: List[str] = []
    if model_dir:
        local_candidates.append(model_dir)
        refs_main = Path(model_dir) / "refs" / "main"
        snapshots_dir = Path(model_dir) / "snapshots"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            local_candidates.insert(0, str(snapshots_dir / revision))
        elif snapshots_dir.exists():
            local_candidates.extend(str(path) for path in sorted(snapshots_dir.iterdir()) if path.is_dir())

    errors: List[str] = []
    for candidate in [*local_candidates, model_name]:
        if not candidate:
            continue
        try:
            print(f"Loading VGGT weights from: {candidate}")
            return VGGT.from_pretrained(candidate, local_files_only=True).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{candidate}: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Failed to load VGGT locally. Tried:\n{joined_errors}")


def run_vggt(image_paths: List[str], model, device: str, mode: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images(image_paths, mode=mode).to(device)
    print(f"Loaded {len(image_paths)} images; VGGT input tensor: {tuple(images.shape)}")

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    num_images = len(image_paths)
    pred_np: Dict[str, np.ndarray] = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            pred_np[key] = squeeze_scene_tensor(value, num_images, key).detach().float().cpu().numpy()

    if "world_points" not in pred_np:
        depth = pred_np["depth"]
        pred_np["world_points"] = unproject_depth_map_to_point_map(depth, pred_np["extrinsic"], pred_np["intrinsic"])
        if "world_points_conf" not in pred_np and "depth_conf" in pred_np:
            pred_np["world_points_conf"] = pred_np["depth_conf"]

    return pred_np, tensor_images_to_uint8(images)


def flatten_conf(conf: np.ndarray) -> np.ndarray:
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    return conf.reshape(-1)


def build_point_cloud(
    predictions: Dict[str, np.ndarray],
    colors: np.ndarray,
    source: str,
    conf_percentile: float,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if source == "depth":
        from vggt.utils.geometry import unproject_depth_map_to_point_map

        points = unproject_depth_map_to_point_map(
            predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
        )
        conf = predictions.get("depth_conf", predictions.get("world_points_conf"))
    else:
        points = predictions["world_points"]
        conf = predictions.get("world_points_conf", predictions.get("depth_conf"))

    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)

    valid = np.isfinite(points_flat).all(axis=1)
    if conf is not None:
        conf_flat = flatten_conf(conf)
        valid &= np.isfinite(conf_flat) & (conf_flat > 1e-5)
        if conf_percentile > 0:
            threshold = np.percentile(conf_flat[valid], conf_percentile) if np.any(valid) else np.inf
            valid &= conf_flat >= threshold

    points_flat = points_flat[valid]
    colors_flat = colors_flat[valid]

    if max_points > 0 and points_flat.shape[0] > max_points:
        rng = np.random.default_rng(0)
        selected = rng.choice(points_flat.shape[0], size=max_points, replace=False)
        points_flat = points_flat[selected]
        colors_flat = colors_flat[selected]

    if points_flat.shape[0] == 0:
        raise RuntimeError("No valid points left after confidence filtering")

    return points_flat.astype(np.float32), colors_flat.astype(np.uint8)


def read_ply_xyz(path: str) -> np.ndarray:
    from plyfile import PlyData

    plydata = PlyData.read(path)
    vertex = plydata.elements[0]
    return np.stack(
        (np.asarray(vertex["x"], dtype=np.float64),
         np.asarray(vertex["y"], dtype=np.float64),
         np.asarray(vertex["z"], dtype=np.float64)),
        axis=1,
    )


def align_points_to_reference(points: np.ndarray, reference_ply: str) -> Tuple[np.ndarray, float]:
    """Uniformly scale/translate VGGT points to the reference scene coordinate range."""
    ref = read_ply_xyz(reference_ply)
    ref = ref[np.isfinite(ref).all(axis=1)]
    src = points[np.isfinite(points).all(axis=1)].astype(np.float64)
    if ref.shape[0] < 10 or src.shape[0] < 10:
        raise RuntimeError("Not enough valid points for alignment")

    ref_center = np.percentile(ref, 50, axis=0)
    src_center = np.percentile(src, 50, axis=0)
    ref_extent = np.percentile(ref, 99, axis=0) - np.percentile(ref, 1, axis=0)
    src_extent = np.percentile(src, 99, axis=0) - np.percentile(src, 1, axis=0)

    valid_axes = src_extent[:2] > 1.0e-8
    if np.any(valid_axes):
        scale = float(np.median(ref_extent[:2][valid_axes] / src_extent[:2][valid_axes]))
    else:
        valid_axes_3d = src_extent > 1.0e-8
        scale = float(np.median(ref_extent[valid_axes_3d] / src_extent[valid_axes_3d]))
    translation = ref_center - src_center * scale
    aligned = points.astype(np.float64) * scale + translation
    print(f"Aligned to reference PLY: {reference_ply}")
    print(f"  scale={scale:.6f}, translation={translation}")
    return aligned.astype(np.float32), scale


def apply_axis_transform(points: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return points
    if transform == "swap_xy_neg_z":
        transformed = points.copy()
        transformed[:, 0] = points[:, 1]
        transformed[:, 1] = points[:, 0]
        transformed[:, 2] = -points[:, 2]
        return transformed
    raise ValueError(f"Unknown axis transform: {transform}")


def write_ascii_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def estimate_gaussian_scale(points: np.ndarray, scale_multiplier: float, sample_size: int = 100_000) -> float:
    """Estimate an isotropic Gaussian scale from nearest-neighbor spacing."""
    if points.shape[0] < 2:
        return 1.0e-3

    sample = points
    if points.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        sample = points[rng.choice(points.shape[0], size=sample_size, replace=False)]

    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(sample).query(sample, k=2, workers=-1)
        spacing = float(np.median(distances[:, 1]))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to estimate NN spacing with scipy ({exc}); using bounding-box fallback")
        extent = np.percentile(sample, 95, axis=0) - np.percentile(sample, 5, axis=0)
        spacing = float(np.linalg.norm(extent) / max(np.cbrt(points.shape[0]), 1.0))

    scale = max(spacing * scale_multiplier, 1.0e-6)
    return scale


def write_gaussian_ply(path: str, points: np.ndarray, colors: np.ndarray, opacity: float, scale_multiplier: float) -> None:
    """Write a minimal SH-degree-0 3DGS PLY compatible with render_video_from_ply.py."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors_float = torch.from_numpy(colors.astype(np.float32) / 255.0)
    features_dc = RGB2SH(colors_float).numpy().astype(np.float32)

    opacity = float(np.clip(opacity, 1.0e-4, 1.0 - 1.0e-4))
    opacity_logit = float(np.log(opacity / (1.0 - opacity)))
    log_scale = float(np.log(estimate_gaussian_scale(points, scale_multiplier)))

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        for name in (
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ):
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        for xyz, sh in zip(points, features_dc):
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"0.0 0.0 0.0 "
                f"{sh[0]:.8f} {sh[1]:.8f} {sh[2]:.8f} "
                f"{opacity_logit:.8f} "
                f"{log_scale:.8f} {log_scale:.8f} {log_scale:.8f} "
                f"1.0 0.0 0.0 0.0\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a colored PLY point cloud from raw images using local VGGT.")
    parser.add_argument(
        "--image_dir",
        default="data/datasets_JAX/JAX_068/images",
        help="Folder containing input images. Default: data/datasets_JAX/JAX_068/images",
    )
    parser.add_argument(
        "--output",
        default="outputs/JAX_068_vggt_points.ply",
        help="Output PLY path. Default: outputs/JAX_068_vggt_points.ply",
    )
    parser.add_argument("--model_name", default="facebook/VGGT-1B", help="Hugging Face model id fallback.")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Local VGGT Hugging Face snapshot/cache folder.")
    parser.add_argument("--mode", choices=("crop", "pad"), default="crop", help="VGGT image preprocessing mode.")
    parser.add_argument("--source", choices=("point", "depth"), default="point", help="Use point-map or depth-unprojection points.")
    parser.add_argument("--conf_percentile", type=float, default=50.0, help="Drop this percentile of low-confidence points.")
    parser.add_argument("--max_points", type=int, default=1_500_000, help="Randomly downsample to at most this many points; <=0 disables.")
    parser.add_argument("--format", choices=("gaussian", "point"), default="gaussian", help="Output PLY format. Use gaussian for render_video_from_ply.py.")
    parser.add_argument("--opacity", type=float, default=0.75, help="Activated Gaussian opacity written to gaussian PLY.")
    parser.add_argument("--scale_multiplier", type=float, default=1.0, help="Multiplier for estimated Gaussian radius.")
    parser.add_argument("--align_reference_ply", default="", help="Reference PLY whose scene scale/center should be matched, e.g. data/datasets_JAX/JAX_068/points3D.ply.")
    parser.add_argument("--axis_transform", choices=("none", "swap_xy_neg_z"), default="none", help="Optional coordinate transform applied before reference alignment.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (very slow).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = os.path.abspath(args.image_dir)
    output = os.path.abspath(args.output)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    image_paths = collect_images(image_dir)
    print(f"Found {len(image_paths)} images in {image_dir}")

    model = load_model(args.model_name, args.model_dir, device)
    predictions, colors = run_vggt(image_paths, model, device, args.mode)
    points, point_colors = build_point_cloud(
        predictions,
        colors,
        source=args.source,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
    )
    points = apply_axis_transform(points, args.axis_transform)
    if args.align_reference_ply:
        points, _ = align_points_to_reference(points, os.path.abspath(args.align_reference_ply))
    if args.format == "gaussian":
        write_gaussian_ply(output, points, point_colors, opacity=args.opacity, scale_multiplier=args.scale_multiplier)
    else:
        write_ascii_ply(output, points, point_colors)
    print(f"Saved {points.shape[0]} colored points to {output}")


if __name__ == "__main__":
    main()
