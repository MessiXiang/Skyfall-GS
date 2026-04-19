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

import math

import torch

from scene import GaussianModel as GaussianModel3D
from scene import GaussianModel2D, create_gaussian_model, create_gaussian_model_from_dataset
from utils.point_utils import depth_to_normal
from utils.sh_utils import eval_sh

try:
    from diff_gauss import GaussianRasterizationSettings as GaussianRasterizationSettings3D
    from diff_gauss import GaussianRasterizer as GaussianRasterizer3D
except ImportError:
    GaussianRasterizationSettings3D = None
    GaussianRasterizer3D = None

try:
    from diff_surfel_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2D
    from diff_surfel_rasterization import GaussianRasterizer as GaussianRasterizer2D
except ImportError:
    GaussianRasterizationSettings2D = None
    GaussianRasterizer2D = None


GaussianModel = GaussianModel3D


def _create_screenspace_points(pc):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def _prepare_colors(viewpoint_camera, pc, pipe, override_color, testing, appearance_embedding):
    means3d = pc.get_xyz
    shs = None
    colors_precomp = None

    embedding = None
    if pc.appearance_enabled:
        if not testing:
            if appearance_embedding is not None:
                embedding = appearance_embedding
            else:
                try:
                    embedding = pc.appearance_embeddings[viewpoint_camera.uid]
                except Exception:
                    print(pc.appearance_embeddings.shape)
                    print("Embedding not found for camera", viewpoint_camera.uid, "use mean embedding instead")
                    with torch.no_grad():
                        embedding = torch.mean(pc.appearance_embeddings, dim=0)
        else:
            with torch.no_grad():
                if appearance_embedding is not None:
                    embedding = appearance_embedding
                else:
                    embedding = torch.mean(pc.appearance_embeddings, dim=0)
                    uid = min(6, len(pc.appearance_embeddings) - 1)
                    embedding = pc.appearance_embeddings[uid]

    if pc.appearance_enabled and embedding is not None:
        embedding_expanded = embedding[None].repeat(len(means3d), 1)
        colors_toned = pc.appearance_mlp(pc._embeddings, embedding_expanded, pc.get_features).clamp_max(1.0)
        shdim = (pc.max_sh_degree + 1) ** 2
        colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
        dir_pp = means3d - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        colors_toned = eval_sh(pc.active_sh_degree, colors_toned, dir_pp_normalized)
        colors_precomp = torch.clamp_min(colors_toned + 0.5, 0.0)
    elif override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = means3d - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return shs, colors_precomp


def _render_3dgs(viewpoint_camera, pc, pipe, bg_color, kernel_size, scaling_modifier=1.0, override_color=None, subpixel_offset=None, testing=False, appearance_embedding=None):
    if GaussianRasterizer3D is None or GaussianRasterizationSettings3D is None:
        raise ImportError("3DGS rasterizer is not installed. Install submodules/diff-gaussian-rasterization-depth first.")

    screenspace_points = _create_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros(
            (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
            dtype=torch.float32,
            device="cuda",
        )

    raster_settings = GaussianRasterizationSettings3D(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer3D(raster_settings=raster_settings)

    means3d = pc.get_xyz
    opacity = pc.get_opacity_with_3D_filter
    scales = None
    rotations = None
    cov3d_precomp = None
    if pipe.compute_cov3D_python:
        cov3d_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    shs, colors_precomp = _prepare_colors(viewpoint_camera, pc, pipe, override_color, testing, appearance_embedding)

    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D=means3d,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=(scales.float() if scales is not None else None),
        rotations=rotations,
        cov3Ds_precomp=cov3d_precomp,
    )

    return {
        "render": rendered_image,
        "render_depth": rendered_depth,
        "render_norm": rendered_norm,
        "render_alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "extra": extra,
    }


def _render_2dgs(viewpoint_camera, pc, pipe, bg_color, kernel_size, scaling_modifier=1.0, override_color=None, subpixel_offset=None, testing=False, appearance_embedding=None):
    del kernel_size, subpixel_offset
    if GaussianRasterizer2D is None or GaussianRasterizationSettings2D is None:
        raise ImportError(
            "2DGS backend requires diff_surfel_rasterization. Install it with `pip install --no-build-isolation <path-to-diff-surfel-rasterization>`."
        )

    screenspace_points = _create_screenspace_points(pc)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings2D(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer2D(raster_settings=raster_settings)

    means3d = pc.get_xyz
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3d_precomp = None
    if pipe.compute_cov3D_python:
        splat2world = pc.get_covariance(scaling_modifier)
        width = viewpoint_camera.image_width
        height = viewpoint_camera.image_height
        near = viewpoint_camera.znear
        far = viewpoint_camera.zfar
        ndc2pix = torch.tensor(
            [
                [width / 2, 0, 0, (width - 1) / 2],
                [0, height / 2, 0, (height - 1) / 2],
                [0, 0, far - near, near],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            device="cuda",
        ).T
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3d_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs, colors_precomp = _prepare_colors(viewpoint_camera, pc, pipe, override_color, testing, appearance_embedding)

    rendered_image, radii, allmap = rasterizer(
        means3D=means3d,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity.float(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3d_precomp,
    )

    render_alpha = allmap[1:2]
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3, :3].T).permute(2, 0, 1)
    render_depth_median = torch.nan_to_num(allmap[5:6], 0, 0)
    render_depth_expected = torch.nan_to_num(allmap[0:1] / render_alpha.clamp_min(1e-8), 0, 0)
    render_dist = allmap[6:7]
    depth_ratio = float(getattr(pipe, "depth_ratio", 0.0))
    surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth).permute(2, 0, 1)
    surf_normal = surf_normal * render_alpha.detach()

    extra = {
        "backend": "2dgs",
        "render_depth_expected": render_depth_expected,
        "render_depth_median": render_depth_median,
        "render_dist": render_dist,
        "surf_normal": surf_normal,
        "supports_surface_regularization": True,
    }
    return {
        "render": rendered_image,
        "render_depth": surf_depth,
        "render_norm": render_normal,
        "render_alpha": render_alpha,
        "render_dist": render_dist,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
        "rend_alpha": render_alpha,
        "rend_normal": render_normal,
        "rend_dist": render_dist,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "extra": extra,
    }


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, kernel_size: float, scaling_modifier=1.0, override_color=None, subpixel_offset=None, testing=False, appearance_embedding=None):
    """Render the scene using the backend selected by the Gaussian model."""
    backend = getattr(pc, "gs_backend", "3dgs")
    if backend == "2dgs":
        return _render_2dgs(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            kernel_size,
            scaling_modifier=scaling_modifier,
            override_color=override_color,
            subpixel_offset=subpixel_offset,
            testing=testing,
            appearance_embedding=appearance_embedding,
        )
    return _render_3dgs(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        kernel_size,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        subpixel_offset=subpixel_offset,
        testing=testing,
        appearance_embedding=appearance_embedding,
    )
