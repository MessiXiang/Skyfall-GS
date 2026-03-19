import os
import sys
import importlib
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage
from torchvision import transforms as tvt

from submodules.MoGe.idu_depth import MoGeIDU

sys.path.append("submodules/vggt")


class VGGTIDU:
    def __init__(
        self,
        save_path: str,
        device: str,
        fov_x: float = 60.0,
        model_name: str = "facebook/VGGT-1B",
        input_size: int = 518,
    ):
        self.save_path = save_path
        self.device = device
        self.fov_x = fov_x
        self.model_name = model_name
        self.input_size = input_size

        os.makedirs(save_path, exist_ok=True)

        # Delay heavy import until needed
        vggt_module = importlib.import_module("vggt.models.vggt")
        VGGT = getattr(vggt_module, "VGGT")
        self.model = VGGT.from_pretrained(model_name).to(device).eval()

    def __del__(self):
        if getattr(self, "model", None) is not None:
            try:
                self.model.to("cpu")
                del self.model
            except Exception as e:
                print(f"Error during VGGT cleanup: {e}")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, refined_imgs: List[PILImage], pbar: bool = True) -> List[np.ndarray]:
        if len(refined_imgs) == 0:
            return []

        images, original_sizes = self._preprocess(refined_imgs)

        use_cuda_autocast = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda_autocast:
            cc = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions = self.model(images.to(self.device))
        else:
            predictions = self.model(images.to(self.device))

        depth = predictions["depth"]
        depth = self._normalize_depth_shape(depth, len(original_sizes))

        resized_depths = []
        for idx, (h, w) in enumerate(original_sizes):
            d = depth[idx].detach().float().cpu()
            d = torch.nn.functional.interpolate(
                d[None, None],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            resized_depths.append(d.numpy())

        return resized_depths

    def _preprocess(self, imgs: List[PILImage]):
        to_tensor = tvt.ToTensor()
        tensors = []
        original_sizes = []  # (H, W)

        for img in imgs:
            if not isinstance(img, Image.Image):
                raise TypeError("VGGTIDU expects PIL images")

            rgb = img.convert("RGB")
            w, h = rgb.size
            original_sizes.append((h, w))

            rgb = rgb.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
            tensors.append(to_tensor(rgb))

        images = torch.stack(tensors, dim=0)
        return images, original_sizes

    @staticmethod
    def _normalize_depth_shape(depth: torch.Tensor, num_imgs: int) -> torch.Tensor:
        # Typical depth shapes:
        # [B, S, H, W, 1] or [S, H, W, 1] or [S, H, W]
        if depth.dim() == 5 and depth.shape[0] == 1:
            depth = depth[0]

        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

        if depth.dim() != 3:
            raise RuntimeError(f"Unexpected VGGT depth shape: {tuple(depth.shape)}")

        if depth.shape[0] != num_imgs:
            raise RuntimeError(
                f"Depth/image count mismatch: depth has {depth.shape[0]} items, expected {num_imgs}"
            )

        return depth


def build_depth_estimator(
    estimator_name: str,
    save_path: str,
    device: str,
    fov_x: float,
    vggt_model_name: str = "facebook/VGGT-1B",
):
    name = estimator_name.lower()
    if name == "moge":
        return MoGeIDU(save_path, device, fov_x)
    if name == "vggt":
        return VGGTIDU(save_path, device, fov_x=fov_x, model_name=vggt_model_name)

    raise ValueError(f"Unknown depth estimator: {estimator_name}. Expected one of [moge, vggt].")
