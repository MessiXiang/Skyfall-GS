#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

HF_HOME_DIR = "/root/autodl-tmp/huggingface"
HF_HUB_CACHE_DIR = "/root/autodl-tmp/huggingface/hub"
os.makedirs(HF_HUB_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", HF_HOME_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HUB_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HUB_CACHE_DIR)

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from random import randint
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, IDUParams

from utils.camera_utils import gen_idu_orbit_camera, cameraList_from_camInfos
from scene.dataset_readers import CameraInfo

from PIL import Image
from utils.idu_depth_utils import build_depth_estimator
from utils.idu_sr_utils import build_super_resolution_processor
from utils.idu_segformer_utils import (
    LoveDASegFormer,
    build_adaptive_targets_from_segmentation,
    refine_loveda_segmentation_with_image,
    save_adaptive_targets_overlay,
    save_segmentation_map,
    save_segmentation_overlay,
    summarize_segmentation,
    write_adaptive_targets_csv,
)

# fused SSIM, for faster training

from fused_ssim import fused_ssim

# from utils.gpu_utils import GPUManager

import lpips
import math

from torchvision.transforms.functional import to_pil_image

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.makedirs("./depth_tmp", exist_ok=True)

def _idu_target_xyz(target_entry):
    if isinstance(target_entry, dict):
        return target_entry["target"]
    return target_entry

def _idu_target_radius(base_radius, target_entry):
    if isinstance(target_entry, dict):
        return float(base_radius) * float(target_entry.get("radius_scale", 1.0))
    return float(base_radius)

def _idu_target_azimuth(target_entry):
    if isinstance(target_entry, dict):
        return target_entry.get("azimuth", None)
    return None

def _parse_adaptive_building_direction_azimuths(value):
    if isinstance(value, (list, tuple)):
        return [float(v) % 360.0 for v in value]
    return [float(v.strip()) % 360.0 for v in str(value).split(",") if v.strip()]

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, num_train_cameras=len(scene.getTrainCameras()))
    if checkpoint:
        print("Restoring model from checkpoint")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # set correct xyz lr scheduler
        opt.position_lr_max_steps = opt.iterations
        opt.densify_until_iter = opt.iterations
        opt.densify_from_iter = 0
        gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
                                                        lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
                                                        lr_delay_mult=opt.position_lr_delay_mult,
                                                        max_steps=opt.position_lr_max_steps)
        print("Restored model from checkpoint at iteration {}".format(first_iter))


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras

    num_train_cams = len(trainCameras)

    depth_estimator_standalone = build_depth_estimator(
        opt.idu_depth_estimator,
        "./depth_tmp",
        "cuda:0",
        60.0,
        vggt_model_name=opt.idu_vggt_model_name,
    )
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

    viewpoint_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        if opt.use_lpips_loss:
            lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0,  gt_image.unsqueeze(0)*2.0-1.0).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
        else:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        depth_loss = 0.0
        if opt.lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth[nan_inf_mask] = 0.0
            gt_depth[nan_inf_mask] = 0.0
            depth_loss += depth_loss_func(gt_depth, depth)

            loss += opt.lambda_depth * depth_loss
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use cross entropy loss
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.nn.functional.binary_cross_entropy(opacity, opacity)
            # opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            loss += opt.lambda_opacity * opacity_loss


        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0 and iteration > opt.start_sample_pseudo and iteration < opt.end_sample_pseudo:
            if not pseudo_stack:
                # sample elevation from 80 to 45
                elevation = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (80 - 45) + 45
                # For Satellite
                radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (300 - 250) + 250
                # For GES
                # radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (100 - 50) + 50
                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, elevation, radius, target_std=opt.target_std)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            pseudo_depth = depth_estimator_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(pseudo_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - args.start_sample_pseudo) / 500., 1)
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (opt.lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
                    if origin_lambda_opacity > 0:
                        opt.lambda_opacity = 0.01
                        opacity_cooldown_iter = 500
                        print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")



            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

@torch.no_grad()
def render_idu_set(views, gaussians, pipeline, background, kernel_size, idu_random_ap=False):
    imgs = []
    for view in tqdm(views, desc="IDU Rendering progress"):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=(not idu_random_ap))["render"]
        img = rendering.cpu().numpy().transpose(1, 2, 0)
        imgs.append(img)
    return imgs

@torch.no_grad()
def render_idu_set_with_coverage(views, gaussians, pipeline, background, kernel_size, idu_random_ap=False):
    imgs = []
    stats = []
    for view in tqdm(views, desc="IDU coverage rendering progress"):
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=(not idu_random_ap))
        img = render_pkg["render"].cpu().numpy().transpose(1, 2, 0)
        imgs.append(img)

        alpha = render_pkg.get("render_alpha", None)
        depth = render_pkg.get("render_depth", None)
        if alpha is None:
            alpha_valid_ratio = 1.0
            alpha_mean = 1.0
        else:
            alpha_tensor = alpha.detach().float().clamp(0.0, 1.0)
            alpha_valid_ratio = float((alpha_tensor > 0.15).float().mean().item())
            alpha_mean = float(alpha_tensor.mean().item())
        if depth is None:
            depth_valid_ratio = 1.0
        else:
            depth_tensor = depth.detach().float()
            depth_valid_ratio = float((torch.isfinite(depth_tensor) & (depth_tensor > 0.0)).float().mean().item())
        coverage = alpha_valid_ratio * depth_valid_ratio
        stats.append({
            "alpha_valid_ratio": alpha_valid_ratio,
            "alpha_mean": alpha_mean,
            "depth_valid_ratio": depth_valid_ratio,
            "coverage": coverage,
            "missing": 1.0 - coverage,
        })
    return imgs, stats

def _as_pil_image(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).convert("RGB")

def _score_vggt_low_confidence_candidates(
    imgs,
    save_path,
    fov_x,
    vggt_model_name,
    confidence_percentile=20.0,
    batch_size=4,
    input_size=518,
):
    """Return one low-confidence score per image; larger means more worth refining."""
    confidence_percentile = float(np.clip(confidence_percentile, 0.0, 100.0))
    batch_size = max(1, int(batch_size))
    scorer = build_depth_estimator(
        "vggt",
        save_path,
        device="cuda:0",
        fov_x=fov_x,
        vggt_model_name=vggt_model_name,
        vggt_input_size=input_size,
    )
    scores = []
    for start in tqdm(range(0, len(imgs), batch_size), desc="VGGT confidence scoring"):
        batch_imgs = [_as_pil_image(img) for img in imgs[start:start + batch_size]]
        _, confidences = scorer.run_with_confidence(batch_imgs, pbar=False, return_confidence=True)
        for conf in confidences:
            conf = np.asarray(conf, dtype=np.float32)
            finite = np.isfinite(conf)
            if finite.any():
                low_conf = np.nanpercentile(conf[finite], confidence_percentile)
                scores.append(float(-low_conf))
            else:
                scores.append(float("inf"))
    del scorer
    torch.cuda.empty_cache()
    return scores

def _select_vggt_guided_idu_views(
    idu_cam_infos,
    imgs,
    dataset,
    elevation,
    radius,
    fov_x,
    vggt_model_name,
    keep_ratio,
    min_keep,
    confidence_percentile,
    batch_size,
    input_size,
):
    if len(idu_cam_infos) == 0:
        return idu_cam_infos, imgs

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    keep_count = int(math.ceil(len(idu_cam_infos) * keep_ratio))
    keep_count = max(1, min(len(idu_cam_infos), max(int(min_keep), keep_count)))
    score_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "vggt_confidence")
    os.makedirs(score_path, exist_ok=True)
    scores = _score_vggt_low_confidence_candidates(
        imgs,
        score_path,
        fov_x,
        vggt_model_name,
        confidence_percentile=confidence_percentile,
        batch_size=batch_size,
        input_size=input_size,
    )
    selected_indices = sorted(np.argsort(scores)[-keep_count:].tolist())
    score_log_path = os.path.join(score_path, "selected_candidates.csv")
    with open(score_log_path, "w") as f:
        f.write("idx,score,selected,image_name\n")
        selected_set = set(selected_indices)
        for idx, (score, cam_info) in enumerate(zip(scores, idu_cam_infos)):
            f.write(f"{idx},{score},{int(idx in selected_set)},{cam_info.image_name}\n")
    print(
        f"VGGT-guided IDU sampling selected {len(selected_indices)}/{len(idu_cam_infos)} "
        f"lowest-confidence candidate views. Log: {score_log_path}"
    )
    return [idu_cam_infos[i] for i in selected_indices], [imgs[i] for i in selected_indices]

def _make_unique_idu_image_names(idu_cam_infos, elevation, radius):
    unique_cam_infos = []
    for idx, cam_info in enumerate(idu_cam_infos):
        unique_cam_infos.append(
            cam_info._replace(image_name=f"e{elevation}_r{radius}_cand{idx:05d}.png")
        )
    return unique_cam_infos

def _laplacian_sharpness(img):
    img = np.asarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img
    gray_t = torch.from_numpy(gray).float()[None, None]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )[None, None]
    lap = torch.nn.functional.conv2d(gray_t, kernel, padding=1)
    return float(lap.var().item())

def _build_knee_candidate_elevations(base_elevation, candidate_range, candidate_step, min_elevation, max_elevation):
    candidate_step = max(1.0e-6, float(candidate_step))
    offsets = np.arange(-float(candidate_range), float(candidate_range) + 0.5 * candidate_step, candidate_step)
    candidates = sorted(
        {
            round(float(np.clip(base_elevation + offset, min_elevation, max_elevation)), 4)
            for offset in offsets
        },
        reverse=True,
    )
    return candidates if candidates else [float(base_elevation)]

def _build_global_knee_candidate_elevations(candidate_step, min_elevation, max_elevation):
    candidate_step = max(1.0e-6, float(candidate_step))
    candidates = sorted(
        {
            round(float(ele), 4)
            for ele in np.arange(float(min_elevation), float(max_elevation) + 0.5 * candidate_step, candidate_step)
            if float(min_elevation) <= float(ele) <= float(max_elevation)
        },
        reverse=True,
    )
    return candidates if candidates else [float(max_elevation)]

def _select_knee_elevation_for_target(
    target,
    target_entry,
    base_elevation,
    radius,
    candidate_elevations,
    num_cams,
    height,
    width,
    fov_x,
    dataset,
    gaussians,
    pipeline,
    background,
    kernel_size,
    idu_random_ap,
    num_train_cams,
    knee_render_size,
    quality_alpha,
    info_beta,
    select_mode,
    quality_threshold,
    aggressive,
    metric_mode,
    missing_penalty,
):
    candidate_infos = []
    target_azimuth = _idu_target_azimuth(target_entry)
    fixed_azimuths = [float(target_azimuth)] if target_azimuth is not None else None
    for ele in candidate_elevations:
        candidate_infos += gen_idu_orbit_camera(
            target,
            ele,
            radius,
            num_cams,
            1,
            knee_render_size,
            knee_render_size,
            fov_x,
            use_new_id=(not idu_random_ap),
            num_train_cams=num_train_cams,
            azimuths=fixed_azimuths,
        )

    candidate_views = cameraList_from_camInfos(candidate_infos, 1, dataset, is_pseudo_cam=idu_random_ap)
    metric_mode = (metric_mode or "coverage").lower()
    if metric_mode == "sharpness":
        candidate_imgs = render_idu_set(candidate_views, gaussians, pipeline, background, kernel_size, idu_random_ap)
        metric_values = np.asarray([_laplacian_sharpness(img) for img in candidate_imgs], dtype=np.float32)
        coverage_values = np.ones_like(metric_values, dtype=np.float32)
        missing_values = np.zeros_like(metric_values, dtype=np.float32)
    else:
        candidate_imgs, coverage_stats = render_idu_set_with_coverage(candidate_views, gaussians, pipeline, background, kernel_size, idu_random_ap)
        coverage_values = np.asarray([item["coverage"] for item in coverage_stats], dtype=np.float32)
        missing_values = np.asarray([item["missing"] for item in coverage_stats], dtype=np.float32)
        metric_values = coverage_values
    max_ele = max(candidate_elevations)
    min_ele = min(candidate_elevations)
    denom = max(max_ele - min_ele, 1.0e-6)
    info_gain = np.asarray([(max_ele - ele) / denom for ele in candidate_elevations], dtype=np.float32)
    info_gain = np.clip(info_gain, 1.0e-3, 1.0)

    mode = (select_mode or "balance").lower()
    metric_matrix = metric_values.reshape(len(candidate_elevations), num_cams)
    coverage_matrix = coverage_values.reshape(len(candidate_elevations), num_cams)
    missing_matrix = missing_values.reshape(len(candidate_elevations), num_cams)
    selected_infos = []
    selected_elevations = []
    score_log = []
    for cam_idx in range(num_cams):
        cam_metric = metric_matrix[:, cam_idx]
        cam_coverage = coverage_matrix[:, cam_idx]
        cam_missing = missing_matrix[:, cam_idx]
        max_metric = max(float(cam_metric.max()), 1.0e-8)
        quality = np.clip(cam_metric / max_metric, 0.0, 1.0)
        scores = (quality ** float(quality_alpha)) * (info_gain ** float(info_beta)) - float(missing_penalty) * cam_missing
        selected_idx = 0
        if mode == "max_drop" and len(candidate_elevations) > 1:
            drops = quality[:-1] - quality[1:]
            drop_idx = int(np.argmax(drops))
            selected_idx = drop_idx + (1 if aggressive else 0)
        elif mode == "threshold":
            below = np.where(quality < float(quality_threshold))[0]
            if len(below) > 0:
                selected_idx = int(below[0] if aggressive else max(0, below[0] - 1))
            else:
                selected_idx = int(np.argmax(scores))
        else:
            selected_idx = int(np.argmax(scores))

        selected_elevation = float(candidate_elevations[selected_idx])
        selected_cam = gen_idu_orbit_camera(
            target,
            selected_elevation,
            radius,
            num_cams,
            1,
            height,
            width,
            fov_x,
            use_new_id=(not idu_random_ap),
            num_train_cams=num_train_cams,
        )[cam_idx]
        if target_azimuth is not None:
            selected_cam = gen_idu_orbit_camera(
                target,
                selected_elevation,
                radius,
                1,
                1,
                height,
                width,
                fov_x,
                use_new_id=(not idu_random_ap),
                num_train_cams=num_train_cams,
                azimuths=[float(target_azimuth)],
            )[0]
        selected_infos.append(selected_cam)
        selected_elevations.append(selected_elevation)
        score_log.append({
            "cam_idx": cam_idx,
            "selected_idx": selected_idx,
            "selected_elevation": selected_elevation,
            "quality": quality.tolist(),
            "metric": cam_metric.tolist(),
            "coverage": cam_coverage.tolist(),
            "missing": cam_missing.tolist(),
            "scores": scores.tolist(),
        })

    return selected_infos, selected_elevations, score_log

def _generate_knee_guided_idu_cameras(
    targets,
    base_elevation,
    radius,
    idu_num_cams,
    idu_num_samples_per_view,
    height,
    width,
    fov_x,
    dataset,
    gaussians,
    pipeline,
    background,
    kernel_size,
    idu_random_ap,
    num_train_cams,
    candidate_range,
    candidate_step,
    use_global_range,
    min_elevation,
    max_elevation,
    knee_render_size,
    quality_alpha,
    info_beta,
    select_mode,
    quality_threshold,
    aggressive,
    metric_mode,
    missing_penalty,
    log_path,
):
    effective_min_elevation = max(float(min_elevation), float(base_elevation) - 20.0)
    if use_global_range:
        candidate_elevations = _build_global_knee_candidate_elevations(
            candidate_step,
            effective_min_elevation,
            max_elevation,
        )
        print(
            f"Knee-guided GLOBAL elevation candidates: {candidate_elevations} "
            f"(min limited to max({min_elevation}, base_elevation - 20) = {effective_min_elevation})"
        )
    else:
        candidate_elevations = _build_knee_candidate_elevations(
            base_elevation,
            candidate_range,
            candidate_step,
            effective_min_elevation,
            max_elevation,
        )
        print(
            f"Knee-guided local elevation candidates around {base_elevation}: {candidate_elevations} "
            f"(min limited to max({min_elevation}, base_elevation - 20) = {effective_min_elevation})"
        )
    idu_cam_infos = []
    selected_elevations = []
    os.makedirs(log_path, exist_ok=True)
    csv_path = os.path.join(log_path, "knee_selected_elevations.csv")
    csv_rows = ["target_idx,cam_idx,base_elevation,selected_elevation,selected_idx,candidate_elevations,quality,coverage,missing,metric,scores\n"]
    for target_idx, target_entry in enumerate(tqdm(targets, desc="Knee-guided elevation selection")):
        target = _idu_target_xyz(target_entry)
        target_radius = _idu_target_radius(radius, target_entry)
        selected_infos, selected_elevations_for_target, score_log = _select_knee_elevation_for_target(
            target,
            target_entry,
            base_elevation,
            target_radius,
            candidate_elevations,
            idu_num_cams,
            height,
            width,
            fov_x,
            dataset,
            gaussians,
            pipeline,
            background,
            kernel_size,
            idu_random_ap,
            num_train_cams,
            knee_render_size,
            quality_alpha,
            info_beta,
            select_mode,
            quality_threshold,
            aggressive,
            metric_mode,
            missing_penalty,
        )
        selected_elevations.extend(selected_elevations_for_target)
        for cam_info in selected_infos:
            for _ in range(max(1, int(idu_num_samples_per_view))):
                idu_cam_infos.append(cam_info)
        print(
            f"Target {target_idx:03d}: selected elevations "
            f"{['%.2f' % e for e in selected_elevations_for_target]}"
        )
        for item in score_log:
            csv_rows.append(
                f"{target_idx},{item['cam_idx']},{base_elevation},{item['selected_elevation']},"
                f"{item['selected_idx']},"
                f"\"{'|'.join('%.4f' % e for e in candidate_elevations)}\","
                f"\"{'|'.join('%.6f' % q for q in item['quality'])}\","
                f"\"{'|'.join('%.6f' % c for c in item['coverage'])}\","
                f"\"{'|'.join('%.6f' % m for m in item['missing'])}\","
                f"\"{'|'.join('%.8f' % s for s in item['metric'])}\","
                f"\"{'|'.join('%.8f' % s for s in item['scores'])}\"\n"
            )
    if selected_elevations:
        print(
            f"Knee-guided elevation summary: min={min(selected_elevations):.2f}, "
            f"max={max(selected_elevations):.2f}, mean={np.mean(selected_elevations):.2f}"
        )
    with open(csv_path, "w") as f:
        f.writelines(csv_rows)
    print(f"Saved knee-guided elevation debug log: {csv_path}")
    return idu_cam_infos

@torch.no_grad()
def generate_idu_training_set(
    dataset : ModelParams,
    checkpoint_path : str,
    pipeline : PipelineParams,
    targets, elevation, radius, idu_num_cams, idu_num_samples_per_view, height=512, width=512, fov_x=60.0,
    num_steps: int=50, strength=0.1, guidance_scale=1, eta=0.5,
    use_flow_edit: bool=False, flow_edit_n_min: int=0, flow_edit_n_max: int=15, flow_edit_n_max_end: int=15, flow_edit_n_avg: int=1, model_type: str="FLUX",
    use_difix3d: bool=False, difix3d_model: str="nvidia/difix", difix3d_steps: int=1, 
    use_dreamscene: bool=False, use_sd21: bool=True,
    depth_estimator_name: str="moge", vggt_model_name: str="facebook/VGGT-1B",
    difix3d_guidance: float=0.0, difix3d_timesteps: list=None, difix3d_use_reference: bool=False,
    difix3d_prompt: str="remove degradation",
    refine=True, idu_no_curriculum=False, idu_random_ap=False,
    vggt_guided_sampling: bool=False, vggt_candidate_multiplier: int=3,
    vggt_keep_ratio: float=0.35, vggt_min_keep: int=4,
    vggt_confidence_percentile: float=20.0, vggt_confidence_batch_size: int=4,
    vggt_confidence_input_size: int=518,
    use_sr: bool=False, sr_method: str="pil", sr_scale: int=2,
    sr_downsample_back: bool=True, sr_save_upscaled: bool=False,
    sr_model_name: str="stabilityai/stable-diffusion-x4-upscaler",
    sr_prompt: str="high resolution satellite image, sharp buildings, crisp roads, realistic details",
    sr_negative_prompt: str="blur, low resolution, artifacts, distorted geometry, text, watermark",
    sr_steps: int=20, sr_guidance_scale: float=0.0, sr_noise_level: int=20,
    sr_tile_size: int=256, sr_tile_overlap: int=32,
    sr_post_sharpen_percent: int=80, sr_post_sharpen_radius: float=0.8,
    sr_post_sharpen_threshold: int=2,
    knee_elevation_sampling: bool=False, knee_candidate_range: float=10.0,
    knee_candidate_step: float=5.0, knee_use_global_range: bool=False,
    knee_quality_alpha: float=1.5,
    knee_info_beta: float=1.0, knee_min_elevation: float=25.0,
    knee_max_elevation: float=89.0, knee_render_size: int=256,
    knee_select_mode: str="balance", knee_quality_threshold: float=0.65,
    knee_aggressive: bool=False,
    knee_metric_mode: str="coverage", knee_missing_penalty: float=0.35
):

    gaussians = GaussianModel(dataset.sh_degree, dataset.appearance_enabled, dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
    print(f"Loading model from checkpoint {checkpoint_path}")
    (model_params, first_iter) = torch.load(checkpoint_path, weights_only=False)
    gaussians.load_from_checkpoints(model_params)
    base_dir = os.path.dirname(checkpoint_path)
    print(base_dir)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=base_dir)

    
    # print(gaussians._xyz.shape)
    # # print Gaussian scale statistics
    # gs_scale = gaussians.get_scaling.max(dim=1).values
    # print("Min: ", gs_scale.min().item())
    # print("Max: ", gs_scale.max().item())
    # print("Mean: ", gs_scale.mean().item())
    # print("Std: ", gs_scale.std().item())
    # print("Median: ", gs_scale.median().item())
    # print("Q99: ", gs_scale.kthvalue(int(0.99 * gs_scale.shape[0]), dim=0).values.item())
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    idu_cam_infos = []
    candidate_multiplier = max(1, int(vggt_candidate_multiplier)) if vggt_guided_sampling else 1
    adaptive_single_camera_targets = any(isinstance(target_entry, dict) and "azimuth" in target_entry for target_entry in targets)
    base_idu_num_cams = 1 if adaptive_single_camera_targets else int(idu_num_cams)
    candidate_idu_num_cams = max(1, base_idu_num_cams * candidate_multiplier)
    candidate_num_samples_per_view = 1 if vggt_guided_sampling else idu_num_samples_per_view
    if knee_elevation_sampling and not isinstance(elevation, list) and not isinstance(radius, list):
        idu_cam_infos = _generate_knee_guided_idu_cameras(
            targets,
            elevation,
            radius,
            candidate_idu_num_cams,
            candidate_num_samples_per_view,
            height,
            width,
            fov_x,
            dataset,
            gaussians,
            pipeline,
            background,
            kernel_size,
            idu_random_ap,
            (len(scene.getTrainCameras()) if idu_random_ap else None),
            knee_candidate_range,
            knee_candidate_step,
            knee_use_global_range,
            knee_min_elevation,
            knee_max_elevation,
            knee_render_size,
            knee_quality_alpha,
            knee_info_beta,
            knee_select_mode,
            knee_quality_threshold,
            knee_aggressive,
            knee_metric_mode,
            knee_missing_penalty,
            os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "knee_elevation"),
        )
    elif isinstance(elevation, list) and isinstance(radius, list):
        assert len(elevation) == len(radius)
        assert idu_no_curriculum, "When using multiple elevations and radii, idu_no_curriculum must be set to True"
        for ele, rad in zip(elevation, radius):
            for target_entry in targets:
                target = _idu_target_xyz(target_entry)
                target_radius = _idu_target_radius(rad, target_entry)
                idu_cam_infos += gen_idu_orbit_camera(
                    target,
                    ele,
                    target_radius,
                    candidate_idu_num_cams,
                    candidate_num_samples_per_view,
                    height,
                    width,
                    fov_x,
                )
        num_cams = len(idu_cam_infos)
        idu_cam_infos = random.sample(idu_cam_infos, num_cams // len(elevation))
        print("Warning! Sampling a subset of cameras for each elevation/radius pair")
    else:
        for target_entry in targets:
            target = _idu_target_xyz(target_entry)
            target_radius = _idu_target_radius(radius, target_entry)
            idu_cam_infos += gen_idu_orbit_camera(
                target,
                elevation,
                target_radius,
                candidate_idu_num_cams,
                candidate_num_samples_per_view,
                height,
                width,
                fov_x,
                use_new_id=(not idu_random_ap),
                num_train_cams=(len(scene.getTrainCameras()) if idu_random_ap else None),
                azimuths=([float(_idu_target_azimuth(target_entry))] if _idu_target_azimuth(target_entry) is not None else None),
            )
    idu_cam_infos = _make_unique_idu_image_names(idu_cam_infos, elevation, radius)
    print(f"Generated {len(idu_cam_infos)} IDU cameras")

    cam_lists = cameraList_from_camInfos(idu_cam_infos, 1, dataset, is_pseudo_cam=idu_random_ap)
    imgs = render_idu_set(cam_lists, gaussians, pipeline, background, kernel_size, idu_random_ap)

    if vggt_guided_sampling:
        idu_cam_infos, imgs = _select_vggt_guided_idu_views(
            idu_cam_infos,
            imgs,
            dataset,
            elevation,
            radius,
            fov_x,
            vggt_model_name,
            keep_ratio=vggt_keep_ratio / candidate_multiplier,
            min_keep=vggt_min_keep,
            confidence_percentile=vggt_confidence_percentile,
            batch_size=vggt_confidence_batch_size,
            input_size=vggt_confidence_input_size,
        )

    # render folder, used to store the unprocessed images
    frames_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render")
    os.makedirs(frames_path, exist_ok=True)
    for idx, img in enumerate(imgs):
        img_path = os.path.join(frames_path, '{0:05d}'.format(idx) + ".png")
        Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(img_path)
    
    # Load 
    refine_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render_refine")
    refine_pipe = None
    
    final_imgs = []
    if refine:
        if use_flow_edit:
            # pip install diffusers==0.30.1 huggingface-hub==0.33.4 transformers==4.46.3 tokenizers==0.20.3 (default)
            from submodules.FlowEdit.idu_refine import FlowEditRefineIDU
            refine_pipe = FlowEditRefineIDU(
                save_path = refine_path,
                device="cuda:0",
                model_type=model_type
            )
            final_imgs = refine_pipe.run(
                imgs,
                n_min=flow_edit_n_min,
                n_max=flow_edit_n_max,
                n_max_end=flow_edit_n_max_end,
                n_avg=flow_edit_n_avg
            )
        elif use_difix3d:
            refine_pipe = Difix3DRefineIDU(
                save_path=refine_path,
                device="cuda:0",
                model_name=difix3d_model,
                use_reference=difix3d_use_reference
            )
            final_imgs = refine_pipe.run(
                imgs,
                prompt=difix3d_prompt,
                num_inference_steps=difix3d_steps,
                timesteps=difix3d_timesteps,
                guidance_scale=difix3d_guidance
            )
        elif use_dreamscene:
            refine_pipe = DreamSceneRefineIDU(
                save_path=refine_path,
                device="cuda:0",
                model="sd21" if use_sd21 else "diffusionsat",
            )
            final_imgs = refine_pipe.run(
                imgs,
            )
        else:
            raise NotImplementedError("DiffusionSat refine is deprecated")
        if refine_pipe:
            del refine_pipe
        torch.cuda.empty_cache()
    else:   
        for img in imgs:
            # from torch tensor to PIL
            final_imgs.append(Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)))

    if use_sr:
        sr_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render_refine_sr")
        sr_processor = build_super_resolution_processor(
            sr_method,
            sr_path,
            scale=sr_scale,
            downsample_back=sr_downsample_back,
            save_upscaled=sr_save_upscaled,
            model_name=sr_model_name,
            device="cuda:0",
            prompt=sr_prompt,
            negative_prompt=sr_negative_prompt,
            num_inference_steps=sr_steps,
            guidance_scale=sr_guidance_scale,
            noise_level=sr_noise_level,
            tile_size=sr_tile_size,
            tile_overlap=sr_tile_overlap,
            post_sharpen_percent=sr_post_sharpen_percent,
            post_sharpen_radius=sr_post_sharpen_radius,
            post_sharpen_threshold=sr_post_sharpen_threshold,
        )
        final_imgs = sr_processor.run(final_imgs)
        print(
            f"Applied IDU SR ({sr_method}, scale={sr_scale}, "
            f"downsample_back={sr_downsample_back}) to {len(final_imgs)} images."
        )


    depth_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render_depth")
    os.makedirs(depth_path, exist_ok=True)
    depth_estimator = build_depth_estimator(
        depth_estimator_name,
        depth_path,
        device="cuda:0",
        fov_x=fov_x,
        vggt_model_name=vggt_model_name,
    )
    depths = depth_estimator.run(final_imgs)


    final_idu_cam_infos = []
    # Save to cam_infos
    repeat_selected_views = max(1, int(idu_num_samples_per_view)) if vggt_guided_sampling else 1
    for idx, cam_info in enumerate(idu_cam_infos):
        for repeat_idx in range(repeat_selected_views):
            image_name = cam_info.image_name if repeat_selected_views == 1 else cam_info.image_name.replace(".png", f"_rep{repeat_idx:02d}.png")
            final_cam_info = CameraInfo(
                uid=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                FovY=cam_info.FovY, FovX=cam_info.FovX, 
                cx=0, cy=0,
                image=final_imgs[idx], image_path=cam_info.image_path,
                image_name=image_name, 
                depth=depths[idx], mask=None,
                width=cam_info.width, height=cam_info.height
            )
            final_idu_cam_infos.append(final_cam_info)

    final_cam_lists = cameraList_from_camInfos(final_idu_cam_infos, 1, dataset, is_idu=True, is_pseudo_cam=idu_random_ap)
        
    del depth_estimator
    del gaussians
    torch.cuda.empty_cache()

    return final_cam_lists

@torch.no_grad()
def generate_pseudo_cams(
    dataset : ModelParams,
    num_cams: int,
    num_train_cams: int,
    elevation: float=80.0,
    radius: float=300.0,
    target_std: float=64.0
):
    idu_cam_infos = []
    for _ in range(num_cams):
        mean = torch.tensor([0., 0.])
        std = torch.tensor([target_std, target_std])
        xy = torch.normal(mean, std)
        z = torch.tensor([0])
        target = torch.cat((xy, z)).tolist()
        gen_cams = gen_idu_orbit_camera(
            target,
            elevation=elevation,
            radius=radius,
            num_cams=12,
            num_samples=1,
            height=1024,
            width=1024,
            fov=60.0,
            use_new_id=False,
            num_train_cams=num_train_cams
        )
        gen_cam = random.choice(gen_cams)
        idu_cam_infos.append(gen_cam)

    print(f"Generated {len(idu_cam_infos)} pseudo cameras with e={elevation:.2f} r={radius:.2f}")

    final_idu_cam_infos = []
    # Save to cam_infos
    for idx, cam_info in enumerate(idu_cam_infos):
        final_cam_info = CameraInfo(
            uid=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FovY=cam_info.FovY, FovX=cam_info.FovX, 
            cx=0, cy=0,
            image=Image.new("1", (cam_info.width, cam_info.height), (0)), image_path=cam_info.image_path,
            image_name=cam_info.image_name, 
            depth=None, mask=None,
            width=cam_info.width, height=cam_info.height
        )
        final_idu_cam_infos.append(final_cam_info)

    final_cam_lists = cameraList_from_camInfos(final_idu_cam_infos, 1, dataset, is_pseudo_cam=True)
    

    return final_cam_lists

def training_idu_episode(
        dataset, opt, pipe, 
        checkpoint_path,
        targets, elevation, radius, fov,
        idu_num_cams, idu_num_samples_per_view
    ):
    # NOTE: generate pose -> render frame -> refined using DiffusionSat -> use MoGe to predict monocular depth
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    # Generate IDU training set
    if not opt.idu_no_curriculum:
        assert isinstance(elevation, float) and isinstance(radius, float)
    else:
        assert isinstance(elevation, list) and isinstance(radius, list), "Elevation and radius should be list when no_curriculum is True"
    
    # Validate refinement method selection
    if opt.idu_use_flow_edit and opt.idu_use_difix3d:
        raise ValueError("Cannot use both FlowEdit and Difix3D simultaneously. Please choose one refinement method.")
    
    if opt.idu_refine and not opt.idu_use_flow_edit and not opt.idu_use_difix3d and not opt.idu_use_dreamscene:
        print("Warning: Refinement is enabled but no refinement method is selected. Defaulting to FlowEdit.")
        opt.idu_use_flow_edit = True

    idu_cam_list = generate_idu_training_set(
        dataset,
        checkpoint_path,
        pipe,
        targets, elevation, radius, idu_num_cams, idu_num_samples_per_view, height=opt.idu_render_size, width=opt.idu_render_size, fov_x=fov, # GES: fov_x = 20.0, satellite: 60.0
        num_steps=opt.idu_ddim_step, strength=opt.idu_ddim_strength,
        guidance_scale=opt.idu_ddim_guidance_scale, eta=opt.idu_ddim_eta,
        use_flow_edit=opt.idu_use_flow_edit, flow_edit_n_min=opt.idu_flow_edit_n_min, flow_edit_n_max=opt.idu_flow_edit_n_max, flow_edit_n_max_end=opt.idu_flow_edit_n_max_end, flow_edit_n_avg=opt.idu_flow_edit_n_avg,
        model_type=opt.idu_model_type,
        use_difix3d=opt.idu_use_difix3d, difix3d_model=opt.idu_difix3d_model, difix3d_steps=opt.idu_difix3d_steps,
        difix3d_guidance=opt.idu_difix3d_guidance, difix3d_timesteps=opt.idu_difix3d_timesteps, 
        difix3d_use_reference=opt.idu_difix3d_use_reference, difix3d_prompt=opt.idu_difix3d_prompt,
        use_dreamscene=opt.idu_use_dreamscene, use_sd21=opt.idu_use_sd21,
        depth_estimator_name=opt.idu_depth_estimator, vggt_model_name=opt.idu_vggt_model_name,
        refine=opt.idu_refine, idu_no_curriculum=opt.idu_no_curriculum, idu_random_ap=opt.idu_random_ap,
        vggt_guided_sampling=opt.idu_vggt_guided_sampling,
        vggt_candidate_multiplier=opt.idu_vggt_candidate_multiplier,
        vggt_keep_ratio=opt.idu_vggt_keep_ratio,
        vggt_min_keep=opt.idu_vggt_min_keep,
        vggt_confidence_percentile=opt.idu_vggt_confidence_percentile,
        vggt_confidence_batch_size=opt.idu_vggt_confidence_batch_size,
        vggt_confidence_input_size=opt.idu_vggt_confidence_input_size,
        use_sr=opt.idu_use_sr,
        sr_method=opt.idu_sr_method,
        sr_scale=opt.idu_sr_scale,
        sr_downsample_back=opt.idu_sr_downsample_back,
        sr_save_upscaled=opt.idu_sr_save_upscaled,
        sr_model_name=opt.idu_sr_model_name,
        sr_prompt=opt.idu_sr_prompt,
        sr_negative_prompt=opt.idu_sr_negative_prompt,
        sr_steps=opt.idu_sr_steps,
        sr_guidance_scale=opt.idu_sr_guidance_scale,
        sr_noise_level=opt.idu_sr_noise_level,
        sr_tile_size=opt.idu_sr_tile_size,
        sr_tile_overlap=opt.idu_sr_tile_overlap,
        sr_post_sharpen_percent=opt.idu_sr_post_sharpen_percent,
        sr_post_sharpen_radius=opt.idu_sr_post_sharpen_radius,
        sr_post_sharpen_threshold=opt.idu_sr_post_sharpen_threshold,
        knee_elevation_sampling=opt.idu_knee_elevation_sampling,
        knee_candidate_range=opt.idu_knee_candidate_range,
        knee_candidate_step=opt.idu_knee_candidate_step,
        knee_use_global_range=opt.idu_knee_use_global_range,
        knee_quality_alpha=opt.idu_knee_quality_alpha,
        knee_info_beta=opt.idu_knee_info_beta,
        knee_min_elevation=opt.idu_knee_min_elevation,
        knee_max_elevation=opt.idu_knee_max_elevation,
        knee_render_size=opt.idu_knee_render_size,
        knee_select_mode=opt.idu_knee_select_mode,
        knee_quality_threshold=opt.idu_knee_quality_threshold,
        knee_aggressive=opt.idu_knee_aggressive,
        knee_metric_mode=opt.idu_knee_metric_mode,
        knee_missing_penalty=opt.idu_knee_missing_penalty,
    )

    # load Gaussians and scene
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    # set IDU cameras
    scene.train_idu_cameras[1.0] = idu_cam_list
    gaussians.training_setup(
        opt,
        num_train_cameras=len(scene.getTrainCameras()),
        from_scratch=False  
        # NOTE: set appearacne lr to zero and set the xyz lr scheduler
    )
    if checkpoint_path:
        print(f"Restoring model from checkpoint {checkpoint_path}")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint_path, weights_only=False)
        gaussians.restore(model_params, opt, iterative_datasets_update=True)
        print("Restored model from checkpoint at iteration {}".format(first_iter))
        opt.iterations = first_iter + opt.idu_episode_iterations  # TODO: make this a parameter
        idu_densify_until_iter = first_iter + opt.idu_densify_until_iter
        assert idu_densify_until_iter < opt.iterations
        print(f"Set iterations to {opt.iterations}, densify until {idu_densify_until_iter}")
    else:
        raise ValueError("Checkpoint is required for iterative datasets update")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    trainIDUCameras = scene.getTrainIDUCameras().copy()
    allCameras = trainCameras + trainIDUCameras + testCameras

    num_train_cams = len(trainCameras)

    depth_estimator_standalone = build_depth_estimator(
        opt.idu_depth_estimator,
        "./depth_tmp",
        "cuda:0",
        fov,
        vggt_model_name=opt.idu_vggt_model_name,
    )
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)

    viewpoint_train_stack = None
    viewpoint_train_idu_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    testing_iterations = [iter for iter in range(first_iter, opt.iterations + 2, opt.idu_testing_interval)][1:] # skip first iter
    if opt.iterations not in testing_iterations:
        testing_iterations.append(opt.iterations)
    checkpoint_iterations = [opt.iterations]


    checkpoint_path = None

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")

        iter_start.record()

        gaussians.update_learning_rate(iteration - first_iter)  # NOTE: modified for IDU

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        idu_viewpoint = None

        if iteration + opt.idu_iter_full_train <= opt.iterations and random.random() < opt.idu_train_ratio:
            idu_viewpoint = True
            if not viewpoint_train_idu_stack:
                viewpoint_train_idu_stack = scene.getTrainIDUCameras().copy()
            viewpoint_cam = viewpoint_train_idu_stack.pop(randint(0, len(viewpoint_train_idu_stack)-1))
            lambda_depth = opt.lambda_depth
        else:
            idu_viewpoint = False
            if not viewpoint_train_stack:
                viewpoint_train_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_train_stack.pop(randint(0, len(viewpoint_train_stack)-1))
            lambda_depth = 0
        

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset,
            testing=(idu_viewpoint and not opt.idu_random_ap)
            # If running iterative datasets update, render image using mean of training embedding
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        loss = None
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        if opt.idu_refine or not idu_viewpoint:
            Ll1 = l1_loss(image, gt_image)
            if opt.use_lpips_loss:
                lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0,  gt_image.unsqueeze(0)*2.0-1.0).mean()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
            else:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            Ll1 = torch.tensor(0.0)


        depth_loss = 0.0
        if lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth = depth[~nan_inf_mask]
            gt_depth = gt_depth[~nan_inf_mask]
            depth_loss += depth_loss_func(gt_depth, depth)
            if torch.isnan(depth_loss).sum() == 0:
                if loss:
                    loss += lambda_depth * depth_loss
                else:
                    loss = lambda_depth * depth_loss
            else:
                depth_loss = 0.0

            # loss += lambda_depth * depth_loss
        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0:
            if not pseudo_stack:
                # sample elevation from 80 to 45
                elevation = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (85 - 45) + 45
                # radius = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (300 - 250) + 250
                radius = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (150 - 75) + 75  # For GES

                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, elevation, radius)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            pseudo_depth = depth_estimator_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(pseudo_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = 1.0
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use cross entropy loss
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.nn.functional.binary_cross_entropy(opacity, opacity)
            # opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            if loss:
                loss += opt.lambda_opacity * opacity_loss
            else:
                loss = opt.lambda_opacity * opacity_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size),
                iterative_datasets_update=True
            )

            # Densification
            if iteration < idu_densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("densification!")
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)

                if (iteration % opt.opacity_reset_interval == 0 and iteration < opt.iterations - 100) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opt.lambda_opacity = 0.0
                    opacity_cooldown_iter = opt.idu_opacity_cooling_iterations
                    print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")

            if iteration % 100 == 0 and iteration > idu_densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
    return checkpoint_path

def training_idu(dataset, opt, pipe, init_checkpoint_path):
    start_checkpoint_path = init_checkpoint_path
    opt.opacity_reset_interval = opt.idu_opacity_reset_interval
    opt.idu_testing_interval = opt.idu_episode_iterations // 4
    opt.idu_position_lr_max_steps = opt.idu_episode_iterations
    # extract idu params
    idu_params: IDUParams = opt.idu_params[opt.datasets_type]
    opt.idu_radius_list = idu_params.radius_list
    opt.idu_elevation_list = idu_params.elevation_list
    opt.idu_fov = idu_params.fov
    print("===== IDU Params =====")
    print(f"Datasets Type: {opt.datasets_type}")
    print(f"Radius List: {opt.idu_radius_list}")
    print(f"Elevation List: {opt.idu_elevation_list}")
    print(f"FOV: {opt.idu_fov}")
    print("======================")
    targets = generate_idu_targets(dataset, opt, pipe, init_checkpoint_path)
    if not opt.idu_no_curriculum:
        
        for radius, elevation in zip(opt.idu_radius_list, opt.idu_elevation_list):
            print(f"Training IDU episode with elevation {elevation} and radius {radius}")
            print(f"# of IDU targets: {len(targets)}")
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=elevation, radius=radius, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view
            )
    else:
        print("===== Disable IDU curriculum learning =====")
        assert opt.idu_episode_iterations == 10000, "IDU episode iterations should be 10000"
        assert opt.idu_densify_until_iter == 9000, "IDU episode iterations should be 9000"
        for _ in range(5):
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=opt.idu_elevation_list, radius=opt.idu_radius_list, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view
            )
        

def generate_regular_idu_targets(opt):
    x = np.linspace(-opt.idu_grid_width / 2, opt.idu_grid_width / 2, opt.idu_grid_size + 2)
    y = np.linspace(-opt.idu_grid_height / 2, opt.idu_grid_height / 2, opt.idu_grid_size + 2)
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x, y)
    targets = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).tolist()
    assert len(targets) == opt.idu_grid_size * opt.idu_grid_size
    return targets

@torch.no_grad()
def render_idu_zenith_overview(dataset, opt, pipe, checkpoint_path):
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim,
    )
    print(f"Loading model for IDU adaptive overview from checkpoint {checkpoint_path}")
    model_params, first_iter = torch.load(checkpoint_path, weights_only=False)
    gaussians.load_from_checkpoints(model_params)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=os.path.dirname(checkpoint_path))
    del scene

    xyz = gaussians.get_xyz.detach()
    if xyz.numel() == 0:
        raise RuntimeError("Cannot render IDU adaptive overview: Gaussian model has no points.")
    mins = xyz.min(dim=0).values.detach().cpu().numpy()
    maxs = xyz.max(dim=0).values.detach().cpu().numpy()
    center = (mins + maxs) * 0.5
    extent_xy = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), 1.0)
    target = [float(center[0]), float(center[1]), float(center[2])]
    radius = max(float(opt.idu_adaptive_overview_radius), extent_xy * float(opt.idu_adaptive_overview_radius_scale))
    fov = float(opt.idu_adaptive_overview_fov)
    overview_cam_info = gen_idu_orbit_camera(
        target,
        elevation=90.0,
        radius=radius,
        num_cams=1,
        num_samples=1,
        height=int(opt.idu_adaptive_seg_render_size),
        width=int(opt.idu_adaptive_seg_render_size),
        fov=fov,
    )[0]
    overview_cam = cameraList_from_camInfos([overview_cam_info], 1, dataset)[0]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    rendered = render(
        overview_cam,
        gaussians,
        pipe,
        background,
        kernel_size=dataset.kernel_size,
        testing=True,
    )["render"]
    image = to_pil_image(rendered.detach().cpu().clamp(0.0, 1.0))
    world_bounds_xy = (float(mins[0]), float(maxs[0]), float(mins[1]), float(maxs[1]))
    del gaussians
    torch.cuda.empty_cache()
    return image, world_bounds_xy

def generate_idu_targets(dataset, opt, pipe, checkpoint_path):
    if not opt.idu_adaptive_segformer_sampling:
        return generate_regular_idu_targets(opt)

    adaptive_dir = os.path.join(dataset.model_path, "idu", "adaptive_segformer")
    os.makedirs(adaptive_dir, exist_ok=True)
    overview_img, world_bounds_xy = render_idu_zenith_overview(dataset, opt, pipe, checkpoint_path)
    overview_path = os.path.join(adaptive_dir, "zenith_overview.png")
    overview_img.save(overview_path)
    print(f"Saved IDU adaptive zenith overview: {overview_path}")
    print(
        "IDU adaptive world bounds xy: "
        f"min_x={world_bounds_xy[0]:.3f}, max_x={world_bounds_xy[1]:.3f}, "
        f"min_y={world_bounds_xy[2]:.3f}, max_y={world_bounds_xy[3]:.3f}"
    )

    segmenter = LoveDASegFormer(opt.idu_segformer_model_name, device="cuda:0")
    seg_map = segmenter.predict(overview_img)
    raw_seg_map = seg_map
    seg_map = refine_loveda_segmentation_with_image(overview_img, raw_seg_map)
    del segmenter
    torch.cuda.empty_cache()

    save_segmentation_map(raw_seg_map, os.path.join(adaptive_dir, "segmentation_loveda_raw.png"))
    save_segmentation_map(seg_map, os.path.join(adaptive_dir, "segmentation_loveda.png"))
    save_segmentation_overlay(overview_img, seg_map, os.path.join(adaptive_dir, "segmentation_overlay.png"))
    seg_summary = summarize_segmentation(seg_map)
    print(f"IDU adaptive SegFormer label pixels: {seg_summary}")
    targets, summary = build_adaptive_targets_from_segmentation(
        seg_map,
        grid_width=opt.idu_grid_width,
        grid_height=opt.idu_grid_height,
        grid_size=opt.idu_grid_size,
        building_subdivisions=opt.idu_adaptive_building_subdivisions,
        other_subdivisions=opt.idu_adaptive_other_subdivisions,
        building_radius_scale=opt.idu_adaptive_building_radius_scale,
        other_radius_scale=opt.idu_adaptive_other_radius_scale,
        world_bounds_xy=world_bounds_xy,
        max_targets=opt.idu_adaptive_max_targets,
        fine_grid_multiplier=opt.idu_adaptive_fine_grid_multiplier,
        fine_grid_size=opt.idu_adaptive_fine_grid_size,
        coverage_cells=opt.idu_adaptive_coverage_cells,
        building_weight=opt.idu_adaptive_building_weight,
        road_weight=opt.idu_adaptive_road_weight,
        wild_weight=opt.idu_adaptive_wild_weight,
        nms_radius_cells=opt.idu_adaptive_nms_radius_cells,
        building_four_direction_views=opt.idu_adaptive_building_four_direction_views,
        building_direction_azimuths=_parse_adaptive_building_direction_azimuths(opt.idu_adaptive_building_direction_azimuths),
    )
    write_adaptive_targets_csv(targets, os.path.join(adaptive_dir, "adaptive_targets.csv"))
    fine_grid_size = max(opt.idu_grid_size, int(opt.idu_adaptive_fine_grid_size))
    save_adaptive_targets_overlay(
        overview_img,
        targets,
        os.path.join(adaptive_dir, "adaptive_targets_overlay.png"),
        grid_size=opt.idu_grid_size,
        fine_grid_size=fine_grid_size,
    )
    print(
        "IDU adaptive SegFormer targets: "
        f"total={len(targets)}, building={summary.get('building', 0)}, "
        f"road={summary.get('road', 0)}, wild={summary.get('wild', 0)}"
    )
    return targets


def depth_loss_func(gt_depth, depth):
    # gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
    # depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return (1 - pearson_corrcoef(gt_depth, depth)).mean()
    # return min(
    #     1 - pearson_corrcoef(gt_depth, depth),
    #     1 - pearson_corrcoef(1 / (gt_depth + 200.), depth)
    # )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def colorize_depth_torch(depth_tensor, mask=None, normalize=True, cmap='Spectral'):
    """
    Colorize depth map using matplotlib colormap, implemented for PyTorch tensors.
    Args:
        depth_tensor: Input depth tensor [B, H, W] or [H, W]
        mask: Optional mask tensor [B, H, W] or [H, W]
        normalize: Whether to normalize the depth values
        cmap: Matplotlib colormap name
    Returns:
        Colored depth tensor [B, 3, H, W] or [3, H, W]
    """

    # Process each item in batch
    # Convert to numpy for matplotlib colormap
    depth = depth_tensor[0].detach().cpu().numpy()
    
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        mask_b = mask[0].detach().cpu().numpy()
        depth = np.where((depth > 0) & mask_b, depth, np.nan)
    
    # Convert to disparity (inverse depth)
    disp = 1 / depth
    
    # Normalize disparity
    if normalize:
        min_disp = np.nanquantile(disp, 0.01)
        max_disp = np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    
    # Apply colormap
    colored = plt.get_cmap(cmap)(1.0 - disp)
    colored = np.nan_to_num(colored, 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]
    
    # Convert back to torch tensor and rearrange dimensions
    colored = torch.from_numpy(colored).float() / 255.0
    colored = colored.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    
    return colored.to(depth_tensor.device)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, iterative_datasets_update=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::4]}]

        if iterative_datasets_update:
            validation_configs.append({'name': 'train_idu', 'cameras' : scene.getTrainIDUCameras()[::3]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, testing=(config['name'] == 'test'))
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = render_pkg["render_depth"]
                    gt_depth = viewpoint.original_depth.to("cuda")
                    mask = viewpoint.original_mask.cuda()
                    depth = mask * depth
                    gt_depth = mask * gt_depth
                    depth_vis = torch.nan_to_num(depth, nan=0, posinf=0, neginf=0)
                    # Colorize depth
                    colored_depth = colorize_depth_torch(
                        depth_vis,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    colored_gt_depth = colorize_depth_torch(
                        mask * gt_depth,
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + f"_view_{viewpoint.image_name}/depth_colored",
                            colored_depth[None],  # Add batch dimension
                            global_step=iteration,
                            dataformats='NCHW'
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), colored_gt_depth[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--iterative_datasets_update", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not args.iterative_datasets_update:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    else:
    # Start running iterative datasets update
        training_idu(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)
    # All done
    print("\nTraining complete.")
