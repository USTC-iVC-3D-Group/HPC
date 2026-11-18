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

"""Camera geometry, image loading and point hierarchy."""
import torch
import math
import numpy as np
from typing import NamedTuple
from utils.common import PILtoTorch

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    from scene.cameras import Camera

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera: "Camera"):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def unique_with_indices(tensor, dim=0,):
    """Return the unique elements of a tensor and their indices."""
    with torch.no_grad():
        unique, inverse_indices, counts = torch.unique(tensor, return_inverse=True, dim=dim, return_counts=True)
        indices = torch.scatter_reduce(
            input=torch.zeros_like(unique[:,0], device=tensor.device, dtype=torch.double),
            src=torch.arange(tensor.size(0), device=tensor.device, dtype=torch.double),
            dim=dim,
            index=inverse_indices,
            reduce="amin",
            include_self=False,
        ).long()
        return unique, inverse_indices, indices, counts

def find_divide_scale(anchor, target_ratio, level_num, bounds, voxel_size):
    x_bounds_min, x_bounds_max = bounds
    scale_upper = ((x_bounds_max - x_bounds_min) / voxel_size).max()

    def binary_search(scale_upper, scale_lower, anchor, target_ratio):
        while True:
            scale = (scale_upper+scale_lower)/2
            anchor_unique = torch.unique(torch.round(anchor/voxel_size/scale), dim=0)*voxel_size*scale
            ratio = anchor_unique.shape[0]/anchor.shape[0]
            if abs(ratio-target_ratio)<0.01 or (scale_upper - scale_lower).abs()<1:
                break
            if ratio < target_ratio:
                scale_upper = scale
            else:
                scale_lower = scale
        return scale, anchor_unique

    anchor_unique = anchor
    scale_list = []
    scale_lower = 1
    for _ in range(level_num-1):
        scale, anchor_unique = binary_search(scale_upper, scale_lower, anchor_unique, target_ratio)
        scale_lower = scale
        scale_list.append(scale.item())
    return scale_list
