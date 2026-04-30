import os
from typing import List

import torch
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage


HF_HOME_DIR = "/root/autodl-tmp/huggingface"
HF_HUB_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_cache():
    os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HOME", HF_HOME_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HUB_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE_DIR)


def _resampling_filter(name: str = "lanczos"):
    name = (name or "lanczos").lower()
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping.get(name, Image.Resampling.LANCZOS)


def _post_downsample_sharpen(
    img: PILImage,
    percent: int = 0,
    radius: float = 0.8,
    threshold: int = 2,
) -> PILImage:
    if percent <= 0:
        return img
    return img.filter(
        ImageFilter.UnsharpMask(
            radius=radius,
            percent=percent,
            threshold=threshold,
        )
    )


class PILSuperResolutionIDU:
    """Dependency-free SR-style post-process for IDU images.

    This implements the safe Scheme-B path:
    original size -> upscale -> mild sharpen -> downsample back to original size.
    It improves perceived sharpness without changing camera/depth dimensions.
    """

    def __init__(
        self,
        save_path: str,
        scale: int = 2,
        downsample_back: bool = True,
        resample: str = "lanczos",
        sharpen_radius: float = 1.2,
        sharpen_percent: int = 120,
        sharpen_threshold: int = 3,
        save_upscaled: bool = False,
        post_sharpen_percent: int = 80,
        post_sharpen_radius: float = 0.8,
        post_sharpen_threshold: int = 2,
    ):
        self.save_path = save_path
        self.scale = max(1, int(scale))
        self.downsample_back = downsample_back
        self.resample = _resampling_filter(resample)
        self.sharpen_radius = sharpen_radius
        self.sharpen_percent = sharpen_percent
        self.sharpen_threshold = sharpen_threshold
        self.save_upscaled = save_upscaled
        self.post_sharpen_percent = post_sharpen_percent
        self.post_sharpen_radius = post_sharpen_radius
        self.post_sharpen_threshold = post_sharpen_threshold

        os.makedirs(save_path, exist_ok=True)
        if save_upscaled:
            os.makedirs(os.path.join(save_path, "upscaled"), exist_ok=True)

    def run(self, imgs: List[PILImage]) -> List[PILImage]:
        sr_imgs = []
        for idx, img in enumerate(imgs):
            rgb = img.convert("RGB")
            orig_size = rgb.size
            up_size = (orig_size[0] * self.scale, orig_size[1] * self.scale)

            if self.scale > 1:
                up_img = rgb.resize(up_size, self.resample)
            else:
                up_img = rgb.copy()

            up_img = up_img.filter(
                ImageFilter.UnsharpMask(
                    radius=self.sharpen_radius,
                    percent=self.sharpen_percent,
                    threshold=self.sharpen_threshold,
                )
            )

            if self.save_upscaled:
                up_img.save(os.path.join(self.save_path, "upscaled", f"{idx:05d}.png"))

            if self.downsample_back and up_img.size != orig_size:
                out_img = up_img.resize(orig_size, self.resample)
                out_img = _post_downsample_sharpen(
                    out_img,
                    self.post_sharpen_percent,
                    self.post_sharpen_radius,
                    self.post_sharpen_threshold,
                )
            else:
                out_img = up_img

            out_img.save(os.path.join(self.save_path, f"{idx:05d}.png"))
            sr_imgs.append(out_img)

        return sr_imgs


class DiffusersSuperResolutionIDU:
    """Real neural super-resolution backend from Hugging Face diffusers.

    Default model is Stable Diffusion x4 upscaler. For IDU Scheme B, the
    generated high-res image is downsampled back to the original camera size.
    """

    def __init__(
        self,
        save_path: str,
        model_name: str = "stabilityai/stable-diffusion-x4-upscaler",
        device: str = "cuda:0",
        prompt: str = "high resolution satellite image, sharp buildings, crisp roads, realistic details",
        negative_prompt: str = "blur, low resolution, artifacts, distorted geometry, text, watermark",
        num_inference_steps: int = 20,
        guidance_scale: float = 0.0,
        noise_level: int = 20,
        downsample_back: bool = True,
        save_upscaled: bool = False,
        tile_size: int = 256,
        tile_overlap: int = 32,
        post_sharpen_percent: int = 0,
        post_sharpen_radius: float = 0.8,
        post_sharpen_threshold: int = 2,
    ):
        self.save_path = save_path
        self.model_name = model_name
        self.device = device
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.noise_level = noise_level
        self.downsample_back = downsample_back
        self.save_upscaled = save_upscaled
        self.tile_size = max(64, int(tile_size))
        self.tile_overlap = max(0, min(int(tile_overlap), self.tile_size // 2 - 1))
        self.post_sharpen_percent = post_sharpen_percent
        self.post_sharpen_radius = post_sharpen_radius
        self.post_sharpen_threshold = post_sharpen_threshold

        os.makedirs(save_path, exist_ok=True)
        if save_upscaled:
            os.makedirs(os.path.join(save_path, "upscaled"), exist_ok=True)

        configure_hf_cache()

        from diffusers import StableDiffusionUpscalePipeline

        print(f"Loading IDU SR model from Hugging Face cache/mirror: {model_name}")
        try:
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
        except Exception as local_error:
            print(
                f"Local SR weights not found or incomplete ({local_error}); "
                f"downloading from {HF_ENDPOINT} to {HF_HUB_CACHE_DIR}"
            )
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )

        self.pipe = self.pipe.to(device)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()

    def __del__(self):
        if getattr(self, "pipe", None) is not None:
            try:
                self.pipe.to("cpu")
                del self.pipe
            except Exception as e:
                print(f"Error during IDU SR cleanup: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self, imgs: List[PILImage]) -> List[PILImage]:
        sr_imgs = []
        for idx, img in enumerate(imgs):
            rgb = img.convert("RGB")
            orig_size = rgb.size
            up_img = self._upscale_image(rgb)

            if self.save_upscaled:
                up_img.save(os.path.join(self.save_path, "upscaled", f"{idx:05d}.png"))

            out_img = up_img.resize(orig_size, Image.Resampling.LANCZOS) if self.downsample_back else up_img
            if self.downsample_back:
                out_img = _post_downsample_sharpen(
                    out_img,
                    self.post_sharpen_percent,
                    self.post_sharpen_radius,
                    self.post_sharpen_threshold,
                )
            out_img.save(os.path.join(self.save_path, f"{idx:05d}.png"))
            sr_imgs.append(out_img)

        return sr_imgs

    def _upscale_image(self, img: PILImage) -> PILImage:
        if img.width <= self.tile_size and img.height <= self.tile_size:
            return self._upscale_tile(img)
        return self._upscale_tiled(img)

    def _upscale_tile(self, tile: PILImage) -> PILImage:
        result = self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            image=tile,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            noise_level=self.noise_level,
        ).images[0]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result.convert("RGB")

    def _upscale_tiled(self, img: PILImage) -> PILImage:
        scale = 4
        out = Image.new("RGB", (img.width * scale, img.height * scale))
        step = self.tile_size - self.tile_overlap
        y_positions = self._tile_positions(img.height, self.tile_size, step)
        x_positions = self._tile_positions(img.width, self.tile_size, step)

        for y in y_positions:
            for x in x_positions:
                box = (x, y, min(x + self.tile_size, img.width), min(y + self.tile_size, img.height))
                tile = img.crop(box)
                tile_up = self._upscale_tile(tile)
                out.paste(tile_up, (x * scale, y * scale))

        return out

    @staticmethod
    def _tile_positions(length: int, tile_size: int, step: int):
        if length <= tile_size:
            return [0]
        positions = list(range(0, max(1, length - tile_size + 1), step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions


def build_super_resolution_processor(
    method: str,
    save_path: str,
    scale: int = 2,
    downsample_back: bool = True,
    save_upscaled: bool = False,
    model_name: str = "stabilityai/stable-diffusion-x4-upscaler",
    device: str = "cuda:0",
    prompt: str = "high resolution satellite image, sharp buildings, crisp roads, realistic details",
    negative_prompt: str = "blur, low resolution, artifacts, distorted geometry, text, watermark",
    num_inference_steps: int = 20,
    guidance_scale: float = 0.0,
    noise_level: int = 20,
    tile_size: int = 256,
    tile_overlap: int = 32,
    post_sharpen_percent: int = 0,
    post_sharpen_radius: float = 0.8,
    post_sharpen_threshold: int = 2,
):
    name = (method or "pil").lower()
    if name in ("pil", "lanczos", "bicubic"):
        resample = "bicubic" if name == "bicubic" else "lanczos"
        return PILSuperResolutionIDU(
            save_path=save_path,
            scale=scale,
            downsample_back=downsample_back,
            resample=resample,
            save_upscaled=save_upscaled,
            post_sharpen_percent=post_sharpen_percent,
            post_sharpen_radius=post_sharpen_radius,
            post_sharpen_threshold=post_sharpen_threshold,
        )

    if name in ("diffusers", "sd-x4", "stable-diffusion-x4"):
        return DiffusersSuperResolutionIDU(
            save_path=save_path,
            model_name=model_name,
            device=device,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            downsample_back=downsample_back,
            save_upscaled=save_upscaled,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            post_sharpen_percent=post_sharpen_percent,
            post_sharpen_radius=post_sharpen_radius,
            post_sharpen_threshold=post_sharpen_threshold,
        )

    raise ValueError(
        f"Unknown IDU SR method: {method}. Currently supported: pil, lanczos, bicubic, diffusers, sd-x4."
    )
