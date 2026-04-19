import torch


def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    width, height = view.image_width, view.image_height
    ndc2pix = torch.tensor(
        [
            [width / 2, 0, 0, width / 2],
            [0, height / 2, 0, height / 2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device="cuda",
    ).T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrinsics = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device="cuda").float(),
        torch.arange(height, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrinsics.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    return depthmap.reshape(-1, 1) * rays_d + rays_o


def depth_to_normal(view, depth):
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output