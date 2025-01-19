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

import torch
import math
from gsplat import rasterization
from scene.gaussian_model import GaussianModel

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
           stage="fine", cam_type=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration

    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    means3D = pc.get_xyz
    opacity = pc._opacity
    shs = pc.get_features

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        colors = override_color  # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features  # [N, K, 3]
        sh_degree = pc.active_sh_degree

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D.to("cuda"), scales.to(
            "cuda"), rotations.to("cuda"), opacity.to("cuda"), shs.to("cuda")
    elif "fine" in stage:

        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales,
                                                                                                 rotations, opacity,
                                                                                                 shs,
                                                                                                 time)
    else:
        raise NotImplementedError

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to("cuda")  # [4, 4]
    render_colors, render_alphas, info = rasterization(
        means=means3D_final,  # [N, 3]
        quats=rotations_final,  # [N, 4]
        scales=scales_final,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
    )
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)  # [N,]

    try:
        info["means2d"].retain_grad()
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": info["depths"].squeeze(0)}
