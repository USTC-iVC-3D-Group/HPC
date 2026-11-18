"""Render decoder outputs with camera JSON; no GT images or dataset loader."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import math

import numpy as np
import torch
from torchvision.utils import save_image

from codec.io import read_json
from scene.gaussian_model import GaussianModel
from scene.cameras import MiniCam
from utils.geometry import getProjectionMatrix
from gaussian_renderer import prefilter_voxel, render

def make_camera(entry):
    width, height = entry['width'], entry['height']
    c2w = np.eye(4)
    c2w[:3, :3] = entry['rotation']
    c2w[:3, 3] = entry['position']
    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device='cuda').T.contiguous()
    fovx = 2 * math.atan(width / (2 * entry['fx']))
    fovy = 2 * math.atan(height / (2 * entry['fy']))
    projection = getProjectionMatrix(.01, 100., fovx, fovy).cuda().T
    return MiniCam(width, height, fovy, fovx, .01, 100., view, view @ projection)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoded', required=True)
    parser.add_argument('--stream', required=True)
    parser.add_argument('--cameras', required=True, help='Training cameras.json or compatible camera list')
    parser.add_argument('--camera-id', type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--white-background', action='store_true')
    args = parser.parse_args()
    manifest = read_json(Path(args.stream) / 'manifest.json')
    cameras = read_json(args.cameras)
    if args.camera_id is not None:
        cameras = [camera for camera in cameras if camera['id'] == args.camera_id]
    pipeline = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)
    bg = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    model = GaussianModel(**manifest['model'], decoded_version=True)
    for frame in manifest['frames']:
        index = frame['frame_idx']
        source = Path(args.decoded) / f'frame{index:06d}'
        model.load_ply_sparse_gaussian(str(source / 'point_cloud.ply'))
        model.load_mlp_checkpoints(str(source / 'networks.pt'))
        model.eval()
        output = Path(args.output) / f'frame{index:06d}'
        output.mkdir(parents=True, exist_ok=True)
        for entry in cameras:
            camera = make_camera(entry)
            mask = prefilter_voxel(camera, model, pipeline, bg)
            image = render(camera, model, pipeline, bg, visible_mask=mask)['render'].clamp(0, 1)
            save_image(image, str(output / f'camera{entry["id"]:03d}.png'))

if __name__ == '__main__':
    main()
