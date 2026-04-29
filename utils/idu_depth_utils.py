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

HF_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_cache():
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)


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
        configure_hf_cache()

        # Delay heavy import until needed
        vggt_module = importlib.import_module("vggt.models.vggt")
        VGGT = getattr(vggt_module, "VGGT")
        try:
            print(f"Loading VGGT from local Hugging Face cache: {HF_CACHE_DIR}")
            self.model = VGGT.from_pretrained(model_name, local_files_only=True).to(device).eval()
        except Exception as local_error:
            print(f"Local VGGT weights not found or incomplete ({local_error}); downloading from {HF_ENDPOINT} to {HF_CACHE_DIR}")
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
        depths, _ = self.run_with_confidence(refined_imgs, pbar=pbar, return_confidence=False)
        return depths

    @torch.no_grad()
    def run_with_confidence(self, refined_imgs: List[PILImage], pbar: bool = True, return_confidence: bool = True):
        if len(refined_imgs) == 0:
            return [], []

        images, original_sizes = self._preprocess(refined_imgs)

        use_cuda_autocast = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda_autocast:
            cc = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                predictions = self.model(images.to(self.device))
        else:
            predictions = self.model(images.to(self.device))

        depth = self._normalize_map_shape(predictions["depth"], len(original_sizes), "depth")
        confidence = self._extract_confidence(predictions, len(original_sizes)) if return_confidence else None

        resized_depths = []
        resized_confidences = []
        for idx, (h, w) in enumerate(original_sizes):
            d = depth[idx].detach().float().cpu()
            d = torch.nn.functional.interpolate(
                d[None, None],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            resized_depths.append(d.numpy())

            if confidence is not None:
                c = confidence[idx].detach().float().cpu()
                c = torch.nn.functional.interpolate(
                    c[None, None],
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                resized_confidences.append(c.numpy())

        return resized_depths, resized_confidences

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

    def _extract_confidence(self, predictions, num_imgs: int):
        # VGGT versions expose confidence under slightly different names.
        for key in ("depth_conf", "depth_confidence", "world_points_conf", "conf", "confidence"):
            if key in predictions:
                return self._normalize_map_shape(predictions[key], num_imgs, key)
        print("Warning: VGGT did not return a confidence map; falling back to inverse depth-gradient confidence.")
        depth = self._normalize_map_shape(predictions["depth"], num_imgs, "depth")
        grad_y = torch.nn.functional.pad(torch.abs(depth[:, 1:, :] - depth[:, :-1, :]), (0, 0, 0, 1))
        grad_x = torch.nn.functional.pad(torch.abs(depth[:, :, 1:] - depth[:, :, :-1]), (0, 1, 0, 0))
        return 1.0 / (grad_x + grad_y + 1.0e-6)

    @staticmethod
    def _normalize_map_shape(value: torch.Tensor, num_imgs: int, name: str) -> torch.Tensor:
        # Typical depth shapes:
        # [B, S, H, W, 1], [B, S, H, W], [S, H, W, 1], [S, H, W]
        if value.dim() == 5 and value.shape[0] == 1:
            value = value[0]

        if value.dim() == 4 and value.shape[0] == 1 and value.shape[1] == num_imgs:
            value = value[0]

        if value.dim() == 4 and value.shape[-1] == 1:
            value = value[..., 0]

        if value.dim() == 4 and value.shape[1] == 1:
            value = value[:, 0]

        if value.dim() == 2:
            value = value.unsqueeze(0)

        if value.dim() != 3:
            raise RuntimeError(f"Unexpected VGGT {name} shape: {tuple(value.shape)}")

        if value.shape[0] != num_imgs:
            raise RuntimeError(
                f"VGGT {name}/image count mismatch: {name} has {value.shape[0]} items, expected {num_imgs}"
            )

        return value


def build_depth_estimator(
    estimator_name: str,
    save_path: str,
    device: str,
    fov_x: float,
    vggt_model_name: str = "facebook/VGGT-1B",
    vggt_input_size: int = 518,
):
    name = estimator_name.lower()
    if name == "moge":
        return MoGeIDU(save_path, device, fov_x)
    if name == "vggt":
        return VGGTIDU(save_path, device, fov_x=fov_x, model_name=vggt_model_name, input_size=vggt_input_size)

    raise ValueError(f"Unknown depth estimator: {estimator_name}. Expected one of [moge, vggt].")
