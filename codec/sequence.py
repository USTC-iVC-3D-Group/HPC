"""Decode without Scene, a dataset, or per-frame training checkpoints."""
from pathlib import Path
import inspect

import torch
from torch import nn

from codec.io import FORMAT, read_json, write_json
from scene.gaussian_model import GaussianModel
from utils.entropy import EntropyBottleneck
from utils.networks import LatentDecoupleNet, PositionBasedAggregation

def model_spec(model):
    names = inspect.signature(GaussianModel.__init__).parameters
    return {name: getattr(model, name) for name in names
            if name not in ('self', 'mode', 'decoded_version', 'ref_path',
                            'ref_iter', 'frame_idx')}

def write_frame_config(model, directory):
    dynamic = model.mode == 'P_frame'
    info = {'format': FORMAT, 'frame_idx': int(model.frame_idx),
            'kind': 'dynamic' if dynamic else 'initial',
            'model': model_spec(model)}
    if dynamic:
        info.update(gop_size=getattr(model, 'gop_size', 5), bitdepth=model.qb,
                    hpc={'level_num': model.level_num,
                         'target_ratio': model.target_ratio,
                         'n_features_ttl': model.latent_dim})
    write_json(Path(directory) / 'frame_config.json', info)

def network_reference(model):
    return {name: {key: value.detach().clone() for key, value in
                   getattr(model, name).state_dict().items()}
            for name in ('decouple_net', 'agg_net')}

def setup_dynamic(model, info, reference):
    """Build architecture/hierarchy using decoded state; never read a checkpoint."""
    cfg = info['hpc']
    model.mode = 'P_frame'
    model.frame_idx = info['frame_idx']
    model.gop_size = info['gop_size']
    model.qb = info['bitdepth']

    model.hpc_q = 1.0
    model.level_num = cfg['level_num']
    model.latent_dim = cfg['n_features_ttl']
    model.target_ratio = cfg['target_ratio']
    model.local_pts = 4
    model.build_hierachy()
    model.latent_num = sum(level.numel() for level in model.latent_list)
    predicted = model.frame_idx > 1 and model.frame_idx % model.gop_size != 0
    model.decouple_net = LatentDecoupleNet(
        model.latent_dim, model.feat_dim, model.n_offsets, qb=model.qb,
        use_ref_params=predicted,
        ref_nn=reference['decouple_net'] if predicted else None).cuda()
    model.agg_net = PositionBasedAggregation(
        K=4, dim=model.latent_dim // model.level_num, qb=model.qb,
        use_ref_params=predicted,
        ref_nn=reference['agg_net'] if predicted else None).cuda()
    model.entropy_model = EntropyBottleneck(
        model.latent_dim // model.level_num, qb=model.qb).cuda()

@torch.no_grad()
def advance_dynamic(model):
    """Same feature, anchor and rotated-offset update as save_ply_for_P_frame."""
    d_feat, d_offset, d_anchor, rotation = model.latent_decouple()
    model._anchor = nn.Parameter((model._anchor + d_anchor).detach())
    model._anchor_feat = nn.Parameter((model._anchor_feat + d_feat).detach())
    offsets = torch.bmm(model._offset + d_offset, rotation.reshape(-1, 3, 3).transpose(1, 2))
    model._offset = nn.Parameter(offsets.detach())
    model.mode = 'I_frame'  # The fully reconstructed representation renders directly.
    model.decoded_version = True

@torch.no_grad()
def representation(model):
    return {key: getattr(model, key).detach().cpu().clone() for key in
            ('_anchor', '_anchor_feat', '_offset', '_scaling', '_opacity', '_rotation')}

@torch.no_grad()
def decode_frames(package):
    """Yield (index, model), with only packaged files and decoded history as inputs."""
    package = Path(package)
    manifest = read_json(package / 'manifest.json')
    model = None
    reference = None
    for entry in manifest['frames']:
        index = entry['frame_idx']
        directory = package / f'frame{index:06d}'
        info = read_json(directory / 'frame_config.json')
        if index == 0:
            model = GaussianModel(**info['model'], mode='I_frame',
                                  decoded_version=True, frame_idx=0)
            model.conduct_decoding(str(directory))
        else:
            setup_dynamic(model, info, reference)
            model.conduct_decoding_for_hpc(str(directory))
            reference = network_reference(model)
            advance_dynamic(model)
        model.eval()
        yield index, model
