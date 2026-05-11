import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

HF_HOME_DIR = "/root/autodl-tmp/huggingface"
HF_HUB_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
HF_ENDPOINT = "https://hf-mirror.com"


LOVEDA_LABEL_NAMES = {
    0: "background",
    1: "building",
    2: "road",
    3: "water",
    4: "barren",
    5: "forest",
    6: "agriculture",
}

LOVEDA_PALETTE = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (255, 255, 255),
    3: (0, 0, 255),
    4: (128, 64, 0),
    5: (0, 128, 0),
    6: (255, 255, 0),
}


WILD_LABELS = {3, 4, 5, 6}
ROAD_LABELS = {2}
BUILDING_LABELS = {1}


def compute_valid_image_mask(image: Image.Image, black_threshold: int = 12) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(rgb > int(black_threshold), axis=-1)


def refine_loveda_segmentation_with_image(
    image: Image.Image,
    seg_map: np.ndarray,
    black_threshold: int = 12,
) -> np.ndarray:
    """Correct obvious domain-shift failures on rendered satellite overviews.

    The LoveDA model can collapse the full valid render footprint to building.
    For IDU target placement we only need a practical building / road / wild map,
    so this refines labels from RGB cues and always masks black render borders.
    """
    rgb_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
    rgb = rgb_u8.astype(np.float32) / 255.0
    raw_valid = seg_map > 0
    raw_valid_ratio = float(raw_valid.mean()) if raw_valid.size > 0 else 0.0
    if 0.05 <= raw_valid_ratio <= 0.95:
        valid = raw_valid
    else:
        valid = compute_valid_image_mask(image, black_threshold=black_threshold)
    refined = np.zeros(seg_map.shape[:2], dtype=np.uint8)
    if not valid.any():
        return refined

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    brightness = (r + g + b) / 3.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    saturation = maxc - minc
    green_excess = g - 0.5 * (r + b)
    red_excess = r - 0.5 * (g + b)
    blue_excess = b - 0.5 * (r + g)

    vegetation = valid & (g > r * 1.03) & (g > b * 1.03) & (green_excess > 0.025)
    water = valid & (b > r * 1.04) & (b > g * 1.02) & (brightness < 0.42)
    shadow = valid & (brightness < 0.20)
    bright_roof = valid & (brightness > 0.62)
    warm_roof = valid & (red_excess > 0.035) & (brightness > 0.28)
    cool_roof = valid & (blue_excess > 0.035) & (brightness > 0.35) & ~water
    roof_like = bright_roof | warm_roof | cool_roof
    road = (
        valid
        & ~roof_like
        & ~vegetation
        & ~water
        & (brightness > 0.30)
        & (brightness < 0.62)
        & (saturation < 0.105)
        & (np.abs(red_excess) < 0.045)
        & (np.abs(green_excess) < 0.045)
        & (np.abs(blue_excess) < 0.045)
    )
    wild = vegetation | water | shadow
    building = valid & ~(road | wild)

    refined[building] = 1
    refined[road] = 2
    refined[water] = 3
    refined[vegetation | shadow] = 5
    return refined


def configure_hf_cache():
    os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HOME", HF_HOME_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HUB_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE_DIR)


class LoveDASegFormer:
    def __init__(self, model_name: str, device: str = "cuda:0"):
        configure_hf_cache()
        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "SegFormer adaptive IDU sampling requires transformers. "
                "Please install the repository requirements first."
            ) from exc

        self.model_name = model_name
        self.device = device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=HF_HUB_CACHE_DIR,
                local_files_only=True,
            )
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                cache_dir=HF_HUB_CACHE_DIR,
                local_files_only=True,
            ).to(self.device).eval()
        except Exception as local_error:
            print(
                f"Local SegFormer weights not found or incomplete ({local_error}); "
                f"downloading from {HF_ENDPOINT} to {HF_HUB_CACHE_DIR}"
            )
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=HF_HUB_CACHE_DIR,
            )
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                model_name,
                cache_dir=HF_HUB_CACHE_DIR,
            ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(rgb.height, rgb.width),
            mode="bilinear",
            align_corners=False,
        )
        return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def save_segmentation_overlay(image: Image.Image, seg_map: np.ndarray, output_path: str, alpha: float = 0.45):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(rgb)
    for label, value in LOVEDA_PALETTE.items():
        color[seg_map == label] = value
    overlay = (rgb * (1.0 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)


def save_segmentation_map(seg_map: np.ndarray, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    color = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for label, value in LOVEDA_PALETTE.items():
        color[seg_map == label] = value
    Image.fromarray(color).save(output_path)


def save_adaptive_targets_overlay(
    image: Image.Image,
    target_entries: List[Dict],
    output_path: str,
    grid_size: int,
    fine_grid_size: int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size

    fine_grid_size = max(1, int(fine_grid_size))
    grid_size = max(1, int(grid_size))
    for i in range(1, fine_grid_size):
        x = i * width / fine_grid_size
        y = i * height / fine_grid_size
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 35), width=1)
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 35), width=1)
    for i in range(1, grid_size):
        x = i * width / grid_size
        y = i * height / grid_size
        draw.line([(x, 0), (x, height)], fill=(0, 255, 255, 120), width=2)
        draw.line([(0, y), (width, y)], fill=(0, 255, 255, 120), width=2)

    target_fill = (255, 40, 40, 220)
    for item in target_entries:
        gx, gy = item.get("grid_xy", [0, 0])
        x0 = gx * width / fine_grid_size
        y0 = gy * height / fine_grid_size
        x1 = (gx + 1) * width / fine_grid_size
        y1 = (gy + 1) * height / fine_grid_size
        fine_xy = item.get("fine_xy")
        if fine_xy is None:
            cx = (x0 + x1) * 0.5
            cy = (y0 + y1) * 0.5
        else:
            cx = float(fine_xy[0]) * width / fine_grid_size
            cy = float(fine_xy[1]) * height / fine_grid_size
        r = max(3, min(width, height) / fine_grid_size * 0.12)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=target_fill)

    canvas.save(output_path)


def _dominant_label(labels: np.ndarray) -> int:
    valid = labels.reshape(-1)
    valid = valid[valid > 0]
    if valid.size == 0:
        return 0
    return int(Counter(valid.tolist()).most_common(1)[0][0])


def classify_region(seg_patch: np.ndarray) -> Tuple[str, Dict[str, float]]:
    total = max(1, int(seg_patch.size))
    building_ratio = float(np.isin(seg_patch, list(BUILDING_LABELS)).sum()) / total
    road_ratio = float(np.isin(seg_patch, list(ROAD_LABELS)).sum()) / total
    wild_ratio = float(np.isin(seg_patch, list(WILD_LABELS)).sum()) / total
    dominant = _dominant_label(seg_patch)

    if building_ratio >= max(road_ratio, wild_ratio) and building_ratio > 0.05:
        region_type = "building"
    elif road_ratio >= wild_ratio and road_ratio > 0.05:
        region_type = "road"
    elif wild_ratio > 0.05:
        region_type = "wild"
    elif dominant in BUILDING_LABELS:
        region_type = "building"
    elif dominant in ROAD_LABELS:
        region_type = "road"
    else:
        region_type = "wild"

    return region_type, {
        "building_ratio": building_ratio,
        "road_ratio": road_ratio,
        "wild_ratio": wild_ratio,
        "dominant_label": float(dominant),
    }


def build_adaptive_targets_from_segmentation(
    seg_map: np.ndarray,
    grid_width: float,
    grid_height: float,
    grid_size: int = 0,
    building_subdivisions: int = 2,
    other_subdivisions: int = 1,
    building_radius_scale: float = 0.85,
    other_radius_scale: float = 1.0,
    world_bounds_xy: Optional[Tuple[float, float, float, float]] = None,
    max_targets: int = 64,
    min_coverage_targets: int = 16,
    building_quota: float = 0.70,
    fine_grid_multiplier: int = 3,
    fine_grid_size: Optional[int] = None,
    coverage_cells: int = 24,
    building_weight: float = 1.0,
    road_weight: float = 0.3,
    wild_weight: float = 0.1,
    nms_radius_cells: int = 12,
    building_four_direction_views: bool = False,
    building_direction_azimuths: Optional[List[float]] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    h, w = seg_map.shape[:2]
    grid_size = max(1, int(grid_size))
    if fine_grid_size is None or int(fine_grid_size) <= 0:
        fine_grid_size = grid_size * max(1, int(fine_grid_multiplier))
    fine_grid_size = max(grid_size, int(fine_grid_size))
    max_targets = max(1, int(max_targets))
    coverage_cells = max(1, int(coverage_cells))
    nms_radius_cells = max(1, int(nms_radius_cells))

    if world_bounds_xy is None:
        min_x = -0.5 * float(grid_width)
        max_x = 0.5 * float(grid_width)
        min_y = -0.5 * float(grid_height)
        max_y = 0.5 * float(grid_height)
    else:
        min_x, max_x, min_y, max_y = [float(v) for v in world_bounds_xy]
        if max_x <= min_x or max_y <= min_y:
            raise ValueError(f"Invalid world_bounds_xy: {world_bounds_xy}")

    labels = np.zeros((fine_grid_size, fine_grid_size), dtype=np.uint8)
    for gy in range(fine_grid_size):
        y0 = int(round(gy * h / fine_grid_size))
        y1 = int(round((gy + 1) * h / fine_grid_size))
        for gx in range(fine_grid_size):
            x0 = int(round(gx * w / fine_grid_size))
            x1 = int(round((gx + 1) * w / fine_grid_size))
            patch = seg_map[y0:y1, x0:x1]
            labels[gy, gx] = _dominant_label(patch)

    weight_map = np.zeros_like(labels, dtype=np.float32)
    weight_map[np.isin(labels, list(BUILDING_LABELS))] = float(building_weight)
    weight_map[np.isin(labels, list(ROAD_LABELS))] = float(road_weight)
    weight_map[np.isin(labels, list(WILD_LABELS))] = float(wild_weight)

    half = max(0, coverage_cells // 2)
    candidates = []
    for gy in range(fine_grid_size):
        for gx in range(fine_grid_size):
            y0 = max(0, gy - half)
            y1 = min(fine_grid_size, gy + half + 1)
            x0 = max(0, gx - half)
            x1 = min(fine_grid_size, gx + half + 1)
            window_labels = labels[y0:y1, x0:x1]
            window_weights = weight_map[y0:y1, x0:x1]
            score = float(window_weights.sum())
            if score <= 0.0:
                continue

            total = max(1, int(window_labels.size))
            ratios = {
                "building_ratio": float(np.isin(window_labels, list(BUILDING_LABELS)).sum()) / total,
                "road_ratio": float(np.isin(window_labels, list(ROAD_LABELS)).sum()) / total,
                "wild_ratio": float(np.isin(window_labels, list(WILD_LABELS)).sum()) / total,
                "dominant_label": float(_dominant_label(window_labels)),
            }
            region_scores = {
                "building": ratios["building_ratio"] * float(building_weight),
                "road": ratios["road_ratio"] * float(road_weight),
                "wild": ratios["wild_ratio"] * float(wild_weight),
            }
            region_type = max(region_scores, key=region_scores.get)
            radius_scale = building_radius_scale if region_type == "building" else other_radius_scale
            px = gx + 0.5
            py = gy + 0.5
            world_x = min_x + (px / fine_grid_size) * (max_x - min_x)
            world_y = max_y - (py / fine_grid_size) * (max_y - min_y)
            center_x = 0.5 * (min_x + max_x)
            center_y = 0.5 * (min_y + max_y)
            candidates.append({
                "target": [float(world_x), float(world_y), 0.0],
                "region_type": region_type,
                "radius_scale": float(radius_scale),
                "azimuth": float((np.degrees(np.arctan2(center_y - world_y, center_x - world_x))) % 360.0),
                "grid_xy": [gx, gy],
                "fine_xy": [float(px), float(py)],
                "ratios": ratios,
                "world_bounds_xy": [min_x, max_x, min_y, max_y],
                "score": float(score),
                "coverage_cells": int(coverage_cells),
            })

    selected = []
    remaining_weight = weight_map.copy()
    while len(selected) < max_targets:
        best_item = None
        best_score = 0.0
        for item in candidates:
            gx, gy = item["grid_xy"]
            y0 = max(0, gy - half)
            y1 = min(fine_grid_size, gy + half + 1)
            x0 = max(0, gx - half)
            x1 = min(fine_grid_size, gx + half + 1)
            gain = float(remaining_weight[y0:y1, x0:x1].sum())
            if gain > best_score:
                best_score = gain
                best_item = item
        if best_item is None or best_score <= 0.0:
            break
        selected_item = dict(best_item)
        selected_item["score"] = float(best_score)
        selected.append(selected_item)
        gx, gy = best_item["grid_xy"]
        y0 = max(0, gy - half)
        y1 = min(fine_grid_size, gy + half + 1)
        x0 = max(0, gx - half)
        x1 = min(fine_grid_size, gx + half + 1)
        remaining_weight[y0:y1, x0:x1] = 0.0
        if nms_radius_cells > 0:
            yy, xx = np.ogrid[:fine_grid_size, :fine_grid_size]
            nms_mask = (xx - gx) * (xx - gx) + (yy - gy) * (yy - gy) <= nms_radius_cells * nms_radius_cells
            remaining_weight[nms_mask] = 0.0

    selected.sort(key=lambda item: (item["grid_xy"][1], item["grid_xy"][0]))
    if building_four_direction_views:
        if building_direction_azimuths is None:
            building_direction_azimuths = [0.0, 90.0, 180.0, 270.0]
        # These azimuths are camera-position azimuths around the building target.
        # gen_idu_orbit_camera places the camera at target + offset(azimuth)
        # and always calls look_at_to_c2w(eye, target), so every expanded view
        # is a camera around the building looking inward at the building.
        direction_azimuths = [float(azimuth) % 360.0 for azimuth in building_direction_azimuths]
        expanded = []
        for parent_idx, item in enumerate(selected):
            if item.get("region_type") != "building":
                expanded.append(item)
                continue
            for direction_idx, azimuth in enumerate(direction_azimuths):
                direction_item = dict(item)
                direction_item["azimuth"] = float(azimuth)
                direction_item["direction_idx"] = int(direction_idx)
                direction_item["direction_azimuth"] = float(azimuth)
                direction_item["parent_target_idx"] = int(parent_idx)
                expanded.append(direction_item)
        selected = expanded

    summary = {"building": 0, "road": 0, "wild": 0}
    for item in selected:
        summary[item["region_type"]] += 1
    targets = selected
    return targets, summary


def write_adaptive_targets_csv(target_entries: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("idx,x,y,z,region_type,radius_scale,azimuth,direction_idx,direction_azimuth,parent_target_idx,grid_x,grid_y,score,coverage_cells,building_ratio,road_ratio,wild_ratio,dominant_label,min_x,max_x,min_y,max_y\n")
        for idx, item in enumerate(target_entries):
            ratios = item.get("ratios", {})
            target = item["target"]
            grid_xy = item.get("grid_xy", [-1, -1])
            bounds = item.get("world_bounds_xy", [0.0, 0.0, 0.0, 0.0])
            f.write(
                f"{idx},{target[0]},{target[1]},{target[2]},"
                f"{item.get('region_type', 'unknown')},{item.get('radius_scale', 1.0)},"
                f"{item.get('azimuth', '')},"
                f"{item.get('direction_idx', '')},"
                f"{item.get('direction_azimuth', '')},"
                f"{item.get('parent_target_idx', '')},"
                f"{grid_xy[0]},{grid_xy[1]},"
                f"{item.get('score', 0.0):.6f},"
                f"{item.get('coverage_cells', 0)},"
                f"{ratios.get('building_ratio', 0.0):.6f},"
                f"{ratios.get('road_ratio', 0.0):.6f},"
                f"{ratios.get('wild_ratio', 0.0):.6f},"
                f"{ratios.get('dominant_label', 0.0):.0f},"
                f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}\n"
            )


def summarize_segmentation(seg_map: np.ndarray) -> Dict[str, int]:
    labels, counts = np.unique(seg_map, return_counts=True)
    return {
        LOVEDA_LABEL_NAMES.get(int(label), str(int(label))): int(count)
        for label, count in zip(labels, counts)
    }
