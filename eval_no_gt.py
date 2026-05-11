"""
No-reference evaluation script for generated/rendered videos or image folders.
Calculates MANIQA, MUSIQ, CLIPIQA, NIQE, and BRISQUE without GT/reference images.

Expected directory structure is consistent with eval.py, except no GT folder is required:

results_eval/data_eval_custom/
├── SCENE_001/
│   ├── method_a/
│   │   ├── *.mp4 or *.png/*.jpg
│   ├── method_b/
│   │   ├── *.mp4 or *.png/*.jpg
│   └── ...
└── SCENE_002/
    └── ...

Usage:
python eval_no_gt.py \
    --data_dir results_eval/data_eval_custom \
    --temp_dir /root/autodl-tmp/temp_dir_no_gt \
    --methods ours_stage1 ours_stage2 ours_adaptive \
    --output_file metrics_results_no_gt.csv \
    --frame_rate 30 \
    --resolution 1024 \
    --batch_size 16

python eval_no_gt.py \
    --data_dir results_eval/data_eval_custom \
    --temp_dir /root/autodl-tmp/temp_dir_no_gt \
    --methods ours_stage1 ours_stage2 \
    --output_file metrics_results_no_gt.csv \
    --frame_rate 24 \
    --no_resize \
    --batch_size 16
"""

import os

# Use hf-mirror and store downloaded Hugging Face models on the data disk.
# Keep token lookup compatible with an existing `hf-cli login` by not changing HF_HOME.
HF_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

import argparse
import csv
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pyiqa

warnings.filterwarnings("ignore")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")

# Higher is better for these metrics. NIQE/BRISQUE are lower-is-better.
NO_REF_METRICS = ["maniqa", "musiq", "clipiqa", "niqe", "brisque"]
LOWER_IS_BETTER = {"niqe", "brisque"}


def extract_frames(video_path, output_dir, frame_rate=1, prefix="", resolution=None):
    """Extract uniformly sampled frames from a video file."""
    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}, FPS: {fps}, Total Frames: {total_frames}")

    if total_frames <= 0:
        video.release()
        return 0

    num_frames = min(int(frame_rate), total_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, endpoint=False, dtype=int)

    frame_count = 0
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = video.read()
        if not ret:
            continue
        if resolution is not None:
            frame = cv2.resize(frame, (resolution, resolution))
        cv2.imwrite(os.path.join(output_dir, f"{prefix}frame_{frame_count:05d}.png"), frame)
        frame_count += 1

    video.release()
    return frame_count


def copy_sampled_images(input_dir, output_dir, frame_rate=24, resolution=None, prefix=""):
    """Copy uniformly sampled images from an image folder."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted([p for p in Path(input_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    total_frames = len(image_paths)
    if total_frames == 0:
        return 0

    num_frames = min(int(frame_rate), total_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, endpoint=False, dtype=int)

    count = 0
    for out_idx, src_idx in enumerate(indices):
        src_path = image_paths[int(src_idx)]
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        if resolution is not None:
            img = cv2.resize(img, (resolution, resolution))
        out_path = Path(output_dir) / f"{prefix}frame_{out_idx:05d}.png"
        cv2.imwrite(str(out_path), img)
        count += 1
    return count


def collect_method_frames(method_dir, output_dir, frame_rate=24, resolution=None):
    """Collect sampled frames from videos or images under one method directory."""
    method_dir = Path(method_dir)
    os.makedirs(output_dir, exist_ok=True)

    videos = sorted([p for p in method_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES])
    images = sorted([p for p in method_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])

    total_frames = 0
    if videos:
        for video_path in videos:
            total_frames += extract_frames(
                video_path,
                output_dir,
                frame_rate=frame_rate,
                prefix=f"{video_path.stem}_",
                resolution=resolution,
            )
    elif images:
        total_frames += copy_sampled_images(
            method_dir,
            output_dir,
            frame_rate=frame_rate,
            resolution=resolution,
            prefix=f"{method_dir.name}_",
        )
    else:
        print(f"Warning: no videos or images found in {method_dir}")

    return total_frames


def preprocess_image_for_iqa(img_bgr):
    """Convert BGR image to RGB tensor in [0, 1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return transforms.ToTensor()(img_pil).unsqueeze(0)


class NoReferenceIQACalculator:
    def __init__(self, device="cuda"):
        self.device = device
        self.metrics = {}

        for metric_name in NO_REF_METRICS:
            try:
                self.metrics[metric_name] = pyiqa.create_metric(metric_name, device=device)
                direction = "lower is better" if metric_name in LOWER_IS_BETTER else "higher is better"
                print(f"✓ {metric_name.upper()} metric loaded ({direction})")
            except Exception as e:
                print(f"✗ Failed to load {metric_name.upper()}: {e}")
                self.metrics[metric_name] = None

    def calculate_image_metrics(self, img_bgr):
        img_tensor = preprocess_image_for_iqa(img_bgr).to(self.device)
        results = {}

        for metric_name, metric in self.metrics.items():
            if metric is None:
                results[metric_name] = None
                continue
            try:
                with torch.no_grad():
                    score = metric(img_tensor)
                results[metric_name] = float(score.item())
            except Exception as e:
                print(f"Error calculating {metric_name.upper()}: {e}")
                results[metric_name] = None

        return results


def evaluate_no_reference_frames(frame_paths, iqa_calc, method_name=""):
    all_metrics = {metric_name: [] for metric_name in NO_REF_METRICS}

    for frame_path in tqdm(frame_paths, desc=f"Computing no-reference IQA for {method_name}"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        metrics = iqa_calc.calculate_image_metrics(img)
        for metric_name, value in metrics.items():
            if value is not None and np.isfinite(value):
                all_metrics[metric_name].append(value)

    results = {}
    for metric_name, values in all_metrics.items():
        if values:
            results[metric_name] = float(np.mean(values))
            results[f"{metric_name}_std"] = float(np.std(values))
        else:
            results[metric_name] = None
            results[f"{metric_name}_std"] = None

    results["num_frames"] = len(frame_paths)
    return results


def main():
    parser = argparse.ArgumentParser(description="No-reference evaluation script for MANIQA, MUSIQ, CLIPIQA, NIQE, and BRISQUE")
    parser.add_argument("--data_dir", type=str, default="data_eval", help="Path to the data directory")
    parser.add_argument("--output_file", type=str, default="metrics_results_no_gt.csv", help="Output CSV file name")
    parser.add_argument("--temp_dir", type=str, default="temp_frames_no_gt", help="Path to the temporary directory for frames")
    parser.add_argument("--frame_rate", type=int, default=24, help="Number of frames/images to sample per video or image folder")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution of sampled frames")
    parser.add_argument("--no_resize", action="store_true", default=False, help="Do not resize sampled frames")
    parser.add_argument("--methods", nargs="+", default=["ours_stage1", "ours_stage2"], help="List of methods to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Kept for CLI compatibility with eval.py; no-reference metrics are evaluated per image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (cuda/cpu)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    temp_dir = Path(args.temp_dir)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")
    iqa_calc = NoReferenceIQACalculator(device=device)

    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    results = []
    resolution = None if args.no_resize else args.resolution

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        print("\n" + "=" * 50)
        print(f"Processing scene: {scene_name}")
        print("=" * 50)

        method_combined_dirs = {}
        try:
            for method in args.methods:
                method_dir = scene_dir / method
                if not method_dir.exists():
                    print(f"Warning: method directory not found: {method_dir}; skipping")
                    continue

                print(f"\nProcessing method: {method}")
                method_combined_dir = temp_dir / f"{scene_name}_{method}_combined"
                os.makedirs(method_combined_dir, exist_ok=True)
                method_combined_dirs[method] = method_combined_dir

                total_method_frames = collect_method_frames(
                    method_dir,
                    method_combined_dir,
                    frame_rate=args.frame_rate,
                    resolution=resolution,
                )
                print(f"Collected a total of {total_method_frames} frames/images for {method}")

                frame_paths = sorted([p for p in method_combined_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
                if not frame_paths:
                    print(f"Warning: no sampled frames available for {method}; skipping")
                    continue

                iqa_metrics = evaluate_no_reference_frames(frame_paths, iqa_calc, method_name=method)

                result = {
                    "scene": scene_name,
                    "method": method,
                    "maniqa": iqa_metrics["maniqa"],
                    "musiq": iqa_metrics["musiq"],
                    "clipiqa": iqa_metrics["clipiqa"],
                    "niqe": iqa_metrics["niqe"],
                    "brisque": iqa_metrics["brisque"],
                    "maniqa_std": iqa_metrics["maniqa_std"],
                    "musiq_std": iqa_metrics["musiq_std"],
                    "clipiqa_std": iqa_metrics["clipiqa_std"],
                    "niqe_std": iqa_metrics["niqe_std"],
                    "brisque_std": iqa_metrics["brisque_std"],
                    "num_frames_evaluated": iqa_metrics["num_frames"],
                }
                results.append(result)

                print(f"Results for {method}:")
                for metric_name in NO_REF_METRICS:
                    value = result[metric_name]
                    direction = "↓" if metric_name in LOWER_IS_BETTER else "↑"
                    if value is not None:
                        print(f"  {metric_name.upper()} {direction}: {value:.4f}")
                    else:
                        print(f"  {metric_name.upper()} {direction}: N/A")
        finally:
            for method_dir in method_combined_dirs.values():
                if method_dir.exists():
                    shutil.rmtree(method_dir)

    if results:
        fieldnames = [
            "scene", "method",
            "maniqa", "musiq", "clipiqa", "niqe", "brisque",
            "maniqa_std", "musiq_std", "clipiqa_std", "niqe_std", "brisque_std",
            "num_frames_evaluated",
        ]
        with open(args.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {args.output_file}")
        print("\nSummary Statistics:")
        for method in args.methods:
            method_results = [r for r in results if r["method"] == method]
            if not method_results:
                continue
            print(f"\n{method}:")
            for metric_name in NO_REF_METRICS:
                values = [r[metric_name] for r in method_results if r[metric_name] is not None]
                direction = "↓" if metric_name in LOWER_IS_BETTER else "↑"
                if values:
                    print(f"  Average {metric_name.upper()} {direction}: {np.mean(values):.4f} ± {np.std(values):.4f}")
                else:
                    print(f"  Average {metric_name.upper()} {direction}: N/A")
    else:
        print("No results to save.")

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
