import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel as GaussianModel3D
from scene.gaussian_model import _get_fourier_features
from simple_knn._C import distCUDA2
from torch import nn
from utils.general_utils import build_rotation, build_scaling_rotation, inverse_sigmoid
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModel2D(GaussianModel3D):
    gs_backend = "2dgs"

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    # get_scaling_with_3D_filter and get_opacity_with_3D_filter are inherited
    # from GaussianModel3D — the parent's prod(dim=1) works for 2D scales too

    def get_covariance(self, scaling_modifier=1):
        scaling = self.get_scaling * scaling_modifier
        scaling_3d = torch.cat([scaling, torch.ones_like(scaling[:, :1])], dim=-1)
        rs = build_scaling_rotation(scaling_3d, self.get_rotation).permute(0, 2, 1)
        transform = torch.zeros((self.get_xyz.shape[0], 4, 4), dtype=torch.float32, device="cuda:0")
        transform[:, :3, :3] = rs
        transform[:, 3, :3] = self.get_xyz
        transform[:, 3, 3] = 1.0
        return transform

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        # Use the same distance-based filter as 3DGS
        super().compute_3D_filter(cameras)
        # Parent computes in float64; rasterizer needs float32
        self.filter_3D = self.filter_3D.float()

    def create_from_pcd(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to("cuda:0")
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to("cuda:0"))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to("cuda:0")
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to("cuda:0")), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda:0")
        opacities = self.inverse_opacity_activation(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device="cuda:0")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda:0")
        self.filter_3D = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda:0")

        if self.appearance_enabled:
            self._embeddings = nn.Parameter(
                torch.zeros((self.get_xyz.shape[0], 6 * self.appearance_n_fourier_freqs), device="cuda:0").requires_grad_(True)
            )
            embeddings = _get_fourier_features(pcd.points, num_features=self.appearance_n_fourier_freqs)
            embeddings.add_(torch.randn_like(embeddings) * 0.0001)
            self._embeddings.data.copy_(embeddings)

    def construct_list_of_attributes(self, exclude_filter=False):
        del exclude_filter
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for index in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attributes.append(f"f_dc_{index}")
        for index in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attributes.append(f"f_rest_{index}")
        attributes.append("opacity")
        for index in range(self._scaling.shape[1]):
            attributes.append(f"scale_{index}")
        for index in range(self._rotation.shape[1]):
            attributes.append(f"rot_{index}")
        return attributes

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def save_fused_ply(self, path, color_mapped=False):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.appearance_enabled and color_mapped:
            uid = min(self.appearance_embeddings.shape[0] - 1, 6)
            embedding = self.appearance_embeddings[uid]
            embedding_expanded = embedding[None].repeat(self._xyz.shape[0], 1)
            colors_toned = self.appearance_mlp(self._embeddings, embedding_expanded, self.get_features).clamp_max(1.0)
            shdim = (self.max_sh_degree + 1) ** 2
            colors_toned = colors_toned.view(-1, shdim, 3).contiguous().clamp_max(1.0)
            f_dc = colors_toned[:, :1, :]
            f_rest = colors_toned[:, 1:, :]
            f_dc = f_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = f_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def reset_opacity(self):
        # Account for 3D filter when resetting opacity (same logic as 3DGS)
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter) * 0.01)

        # Reverse the filter effect to get the raw opacity parameter
        scales = self.get_scaling
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        scales_after_square = scales_square + torch.square(self.filter_3D)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / (det2 + 1e-8))
        opacities_new = opacities_new / coef[..., None].clamp(min=1e-6)
        opacities_new = opacities_new.clamp(min=1e-6, max=1 - 1e-6)
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda name: int(name.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for index, attr_name in enumerate(extra_f_names):
            features_extra[:, index] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda name: int(name.split("_")[-1]))
        if len(scale_names) < 2:
            raise ValueError(f"2DGS PLY requires at least 2 scale channels, got {len(scale_names)} from {path}")
        scales = np.zeros((xyz.shape[0], 2))
        for index, attr_name in enumerate(scale_names[:2]):
            scales[:, index] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda name: int(name.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for index, attr_name in enumerate(rot_names):
            rots[:, index] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda:0").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float32, device="cuda:0").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float32, device="cuda:0").transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device="cuda:0").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device="cuda:0").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device="cuda:0").requires_grad_(True))
        self.filter_3D = torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device="cuda:0")
        self.active_sh_degree = self.max_sh_degree

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda:0")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_embeddings = self._embeddings[selected_pts_mask].repeat(N, 1) if self.appearance_enabled else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_embeddings,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda:0", dtype=torch.bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
        )

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_embeddings = self._embeddings[selected_pts_mask] if self.appearance_enabled else None

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_embeddings,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, extent)
        split = self._xyz.shape[0]

        # Extend filter_3D for newly added points (clone/split grow the point count).
        # New points get filter_3D=0 → coef=1 → no opacity reduction, which is
        # conservative; compute_3D_filter() is called right after to set real values.
        n_current = self._xyz.shape[0]
        n_filter = self.filter_3D.shape[0]
        if n_current > n_filter:
            pad = torch.zeros((n_current - n_filter, 1), dtype=torch.float32, device="cuda:0")
            self.filter_3D = torch.cat([self.filter_3D, pad], dim=0)

        # Use filtered opacity for pruning — small Gaussians whose filter-adjusted
        # opacity is below threshold are floaters that don't contribute to rendering
        prune_mask = (self.get_opacity_with_3D_filter < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # Safety: prevent pruning ALL Gaussians
        if prune_mask.all() and prune_mask.numel() > 0:
            print(f"[WARNING] densify_and_prune would remove ALL {prune_mask.sum().item()} Gaussians, keeping top-opacity ones")
            opacity_values = self.get_opacity_with_3D_filter.squeeze()
            keep_count = max(int(prune_mask.numel() * 0.1), 100)
            _, topk_indices = torch.topk(opacity_values, min(keep_count, opacity_values.numel()))
            prune_mask[topk_indices] = False

        self.prune_points(prune_mask)
        # Trim filter_3D to match after pruning
        self.filter_3D = self.filter_3D[~prune_mask]
        after = self._xyz.shape[0]
        return clone - before, split - clone, split - after

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        gradients = torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += gradients
        if hasattr(self, "xyz_gradient_accum_abs"):
            self.xyz_gradient_accum_abs[update_filter] += gradients
        if hasattr(self, "xyz_gradient_accum_abs_max"):
            self.xyz_gradient_accum_abs_max[update_filter] = torch.max(
                self.xyz_gradient_accum_abs_max[update_filter], gradients
            )
        self.denom[update_filter] += 1