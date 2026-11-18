"""HPC networks, temporal references and parameter quantization/coding."""
import torch
import torch.nn as nn
from utils.entropy import Entropy_model_nnfree
import os
import struct
from pytorch3d.ops import knn_gather
from utils.common import quaternion_to_rotation_matrix

NonNNEMGaussian = Entropy_model_nnfree()

def _write_quant_header(file_name, quant_dict):
    """Serialize the scalar range needed to invert one tensor quantizer."""
    header_name = f'{file_name[:-2]}_quant_header.b'
    q_min = float(quant_dict["min"].reshape(-1)[0].detach().cpu())
    q_scale = float(quant_dict["scale"].reshape(-1)[0].detach().cpu())
    with open(header_name, 'wb') as fout:
        fout.write(struct.pack('>ff', q_min, q_scale))
    return 8 * 8

def _read_quant_header(file_name, symbols):
    header_name = f'{file_name[:-2]}_quant_header.b'
    with open(header_name, 'rb') as fin:
        q_min, q_scale = struct.unpack('>ff', fin.read(8))
    return {
        "quant": symbols,
        "min": torch.tensor(q_min, device=symbols.device, dtype=symbols.dtype),
        "scale": torch.tensor(q_scale, device=symbols.device, dtype=symbols.dtype),
    }

def _write_cache_ratio(file_name, cache_ratio):
    with open(os.path.join(file_name, 'cache_ratio.b'), 'wb') as fout:
        fout.write(struct.pack('>f', float(cache_ratio.detach().cpu().reshape(-1)[0])))
    return 32

def _read_cache_ratio(file_name, module):
    with open(os.path.join(file_name, 'cache_ratio.b'), 'rb') as fin:
        value = struct.unpack('>f', fin.read(4))[0]
    module.cache_ratio.data.fill_(value)

class LinearWithCache(nn.Linear):
    def __init__(self, in_features, out_features, ref_nn, nn_prefix, init_ratio=0.0):
        super().__init__(in_features, out_features)

        if f'{nn_prefix}.cache_ratio' in ref_nn.keys():

            ratio = torch.tanh(ref_nn[f'{nn_prefix}.cache_ratio']) + 1.0

            self.register_buffer('cache_weight', (ref_nn[f'{nn_prefix}.weight'] + ref_nn[f'{nn_prefix}.cache_weight']*ratio).detach()) # P_(t-1)
            self.register_buffer('cache_bias', (ref_nn[f'{nn_prefix}.bias'] + ref_nn[f'{nn_prefix}.cache_bias']*ratio).detach())
        else:

            self.register_buffer('cache_weight', ref_nn[f'{nn_prefix}.weight'].detach())
            self.register_buffer('cache_bias', ref_nn[f'{nn_prefix}.bias'].detach())

        self.cache_ratio = nn.Parameter(torch.tensor([init_ratio]))

    def forward(self, x):
        ratio = torch.tanh(self.cache_ratio) + 1.0

        return nn.functional.linear(x, (self.weight + self.cache_weight*ratio), (self.bias + self.cache_bias*ratio))

class Quantize_MLP_ours(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mlp, bitdepth):
        if getattr(mlp, "codec_frozen", False):
            return mlp
        ref_min = None
        ref_max = None
        for module in mlp.modules():
            if isinstance(module, nn.Linear) or isinstance(module, LinearWithCache):
                weight = module.weight.data
                bias = module.bias.data if module.bias is not None else None
                ref_min = ref_max = None

                if isinstance(module, LinearWithCache):
                    ref_min = module.cache_weight.min()
                    ref_max = module.cache_weight.max()
                _, new_weight = quant_tensor_global(weight, bitdepth, ref_min=ref_min, ref_max=ref_max)
                module.weight.data = new_weight
                if module.bias is not None:

                    if isinstance(module, LinearWithCache):
                        ref_min = module.cache_bias.min()
                        ref_max = module.cache_bias.max()
                    _, new_bias = quant_tensor_global(bias, bitdepth, ref_min=ref_min, ref_max=ref_max)
                    module.bias.data = new_bias
        ctx.save_for_backward(mlp)
        return mlp

    @staticmethod
    def backward(ctx, grad_output):
        mlp, = ctx.saved_tensors
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                module.weight.grad = grad_output * 1 if module.weight.grad is None else module.weight.grad + grad_output * 1
                if module.bias is not None:
                    module.bias.grad = grad_output * 1 if module.bias.grad is None else module.bias.grad + grad_output * 1

        return grad_output, None

def estimate_mlp_bits(mlp, bitdepth):
    bits = 0
    for _, module in enumerate(mlp.modules()):
        if isinstance(module, nn.Linear) or isinstance(module, LinearWithCache):
            original_weight = module.weight
            if isinstance(module, LinearWithCache):
                code = scale_parameters_with_grad(original_weight, bitdepth, ref_min=module.cache_weight.min(), ref_max=module.cache_weight.max())
                bits += NonNNEMGaussian.cal_bitrate(code, training=True)["estimated_bits"]
            else:
                code = scale_parameters_with_grad(original_weight, bitdepth, ref_min=None, ref_max=None)
                bits += NonNNEMGaussian.cal_bitrate(code, training=True)["estimated_bits"]

            if module.bias is not None:
                original_bias = module.bias

                if isinstance(module, LinearWithCache):
                    code = scale_parameters_with_grad(original_bias, bitdepth, ref_min=module.cache_bias.min(), ref_max=module.cache_bias.max())
                    bits += NonNNEMGaussian.cal_bitrate(code, training=True)["estimated_bits"]
                else:
                    code = scale_parameters_with_grad(original_bias, bitdepth, ref_min=None, ref_max=None)
                    bits += NonNNEMGaussian.cal_bitrate(code, training=True)["estimated_bits"]

    return bits

@torch.no_grad()
def encode_mlp(mlp, bitdepth, file_name):
    symbols = {"weight": [], "bias": []}
    bits = 0
    for idx, module in enumerate(mlp.modules()):
        if isinstance(module, nn.Linear) or isinstance(module, LinearWithCache):
            original_weight = module.weight.data
            if isinstance(module, LinearWithCache):
                quant_dict, _ = quant_tensor_global(original_weight, bitdepth, ref_max=module.cache_weight.max(), ref_min=module.cache_weight.min())
            else:
                quant_dict, _ = quant_tensor_global(original_weight, bitdepth)
            weight_name = os.path.join(file_name, f'{str(idx)}_weight.b')
            bits += NonNNEMGaussian.compress(quant_dict["quant"], weight_name)
            bits += _write_quant_header(weight_name, quant_dict)
            symbols["weight"].append(quant_dict)
            if module.bias is not None:
                original_bias = module.bias.data
                if isinstance(module, LinearWithCache):
                    quant_dict, _ = quant_tensor_global(original_bias, bitdepth, ref_max=module.cache_bias.max(), ref_min=module.cache_bias.min())
                else:
                    quant_dict, _ = quant_tensor_global(original_bias, bitdepth)
                bias_name = os.path.join(file_name, f'{str(idx)}_bias.b')
                bits += NonNNEMGaussian.compress(quant_dict["quant"], bias_name)
                bits += _write_quant_header(bias_name, quant_dict)
                symbols["bias"].append(quant_dict)
            if isinstance(module, LinearWithCache):
                ratio_dir = os.path.join(file_name, f'{str(idx)}')
                os.makedirs(ratio_dir, exist_ok=True)
                bits += _write_cache_ratio(ratio_dir, module.cache_ratio)

    return symbols, bits

@torch.no_grad()
def decode_mlp(mlp, bitdepth, file_name):
    for idx, module in enumerate(mlp.modules()):
        if isinstance(module, nn.Linear) or isinstance(module, LinearWithCache):
            weight_name = os.path.join(file_name, f'{str(idx)}_weight.b')
            quant_weight = NonNNEMGaussian.decompress(weight_name).to(module.weight.device)
            quant_weight = quant_weight.view_as(module.weight)
            module.weight.data.copy_(dequant_tensor(_read_quant_header(weight_name, quant_weight)))
            if module.bias is not None:
                bias_name = os.path.join(file_name, f'{str(idx)}_bias.b')
                quant_bias = NonNNEMGaussian.decompress(bias_name).to(module.bias.device)
                quant_bias = quant_bias.view_as(module.bias)
                module.bias.data.copy_(dequant_tensor(_read_quant_header(bias_name, quant_bias)))
            if isinstance(module, LinearWithCache):
                _read_cache_ratio(os.path.join(file_name, f'{str(idx)}'), module)
    return mlp

@torch.no_grad()
def encode_tensor(tensor, bitdepth, file_name):
    quant_dict, _ = quant_tensor_global(tensor, bitdepth)
    bits = NonNNEMGaussian.compress(quant_dict["quant"], file_name)
    bits += _write_quant_header(file_name, quant_dict)
    return quant_dict, bits

@torch.no_grad()
def decode_tensor(file_name):
    symbols = NonNNEMGaussian.decompress(file_name)
    return _read_quant_header(file_name, symbols)

@torch.no_grad()
def load_tensor_symbols_dict(symbols):
    dq = dequant_tensor(symbols)
    return dq

def Quantize_Parameters(p, bitdepth):
    _, dp = quant_tensor_global(p, bitdepth)

    return (dp - p).detach() + p

def scale_parameters_with_grad(t, bits, ref_min=None, ref_max=None):
    t_min, t_max = t.min(), t.max()
    if ref_min is not None:
        t_min = min(ref_min, t.min())
    if ref_max is not None:
        t_max = max(ref_max, t.max())
    scale = ((t_max - t_min) / (2**bits - 1)).detach()
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1e10, neginf=1.0).clamp(1e-12, 1e10)
    t_min, scale = t_min.expand_as(t), scale.expand_as(t)
    scale_t = (t - t_min) / (scale)
    noise = torch.empty_like(scale_t).uniform_(-0.5, 0.5)
    return scale_t + noise

def quant_tensor_global(t, bits=8, *, ref_max=None, ref_min=None):
    t_min, t_max = t.min(), t.max()
    if not torch.isfinite(t_min) or not torch.isfinite(t_max):
        raise ValueError(f'Nonfinite quantizer input, shape={tuple(t.shape)}')
    for ref in (ref_min, ref_max):
        if ref is not None and not torch.isfinite(ref):
            raise ValueError('Nonfinite quantizer reference range')
    if ref_min is not None:
        t_min = min(ref_min, t_min)
    if ref_max is not None:
        t_max = max(ref_max, t_max)
    scale = (t_max - t_min) / (2**bits - 1)
    t_min, scale = t_min.expand_as(t), scale.expand_as(t)
    quant_t = ((t - t_min) / torch.where(scale == 0, torch.ones_like(scale), scale)).round().clamp(0, 2**bits-1)
    new_t = t_min + scale * quant_t
    quant_t = {'quant': quant_t, 'min': t_min, 'scale': scale}

    return quant_t, new_t

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'], quant_t['scale']
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t

class PositionBasedAggregation(nn.Module):
    def __init__(self, K=4, dim=16, qb=8, use_ref_params=False, ref_nn=None):
        super(PositionBasedAggregation, self).__init__()
        self.K = K
        self.dim = dim
        if use_ref_params:

            self.weight_mlp = nn.Sequential(
                LinearWithCache(3, self.dim, ref_nn, nn_prefix='weight_mlp.0'),
                nn.ReLU(True),
                LinearWithCache(self.dim, self.dim, ref_nn, nn_prefix='weight_mlp.2'),
            )
            self.channel_mixer = nn.Sequential(
                LinearWithCache(self.dim, self.dim, ref_nn, nn_prefix='channel_mixer.0'),
                nn.ReLU(True),
                LinearWithCache(self.dim, self.dim, ref_nn, nn_prefix='channel_mixer.2'),
            )
        else:
            self.weight_mlp = nn.Sequential(
                nn.Linear(3, dim),
                nn.ReLU(True),
                nn.Linear(dim, dim),
            )
            self.channel_mixer = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(True),
                nn.Linear(self.dim, self.dim),
            )
        self.qb = qb

    def encode_net(self, file_name):
        params_symbols = {}
        params_symbols["weight_mlp"], bits_weight_mlp = encode_mlp(self.weight_mlp, self.qb, os.path.join(file_name, 'weight_mlp'))
        params_symbols["channel_mixer"], bits_channel_mixer = encode_mlp(self.channel_mixer, self.qb, os.path.join(file_name, 'channel_mixer'))
        total_bits = bits_weight_mlp + bits_channel_mixer
        return params_symbols, total_bits

    def decode_net(self, file_name):
        params_symbols = {}
        params_symbols["weight_mlp"] = decode_mlp(self.weight_mlp, self.qb, os.path.join(file_name, 'weight_mlp'))
        params_symbols["channel_mixer"] = decode_mlp(self.channel_mixer, self.qb, os.path.join(file_name, 'channel_mixer'))

    def estimate_params_bits(self):
        estimated_bits = estimate_mlp_bits(self.weight_mlp, self.qb) + estimate_mlp_bits(self.channel_mixer, self.qb)
        return estimated_bits

    def forward(self, coords, feats, knn_idx):

        if self.qb is not None:
            self.weight_mlp = Quantize_MLP_ours.apply(self.weight_mlp, self.qb)
            self.channel_mixer = Quantize_MLP_ours.apply(self.channel_mixer, self.qb)
        knn_idx = knn_idx.unsqueeze(0)
        knn_coords = knn_gather(coords.unsqueeze(0), knn_idx).squeeze(0)
        relative_pos = knn_coords - coords.unsqueeze(1).expand(-1, self.K, -1)
        knn_feats = knn_gather(feats.unsqueeze(0), knn_idx).squeeze(0)
        weight = self.weight_mlp(relative_pos).view(coords.shape[0], self.K, self.dim)
        weight = torch.softmax(weight, dim=1)

        out = knn_feats * weight
        out = torch.sum(out, dim=1)
        out = out + feats
        out = self.channel_mixer(out) + out

        return out

class LatentDecoupleNet(nn.Module):
    def __init__(self, latent_dim, feat_dim, n_offsets, qb=8, use_ref_params=False, ref_nn=None):
        super(LatentDecoupleNet, self).__init__()
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.d_anchor_factor = 0.05
        self.anchor_rot_factor = 0.1

        if use_ref_params:
            self.ntc_mlp = nn.Sequential(
                LinearWithCache(self.latent_dim, self.feat_dim, ref_nn, nn_prefix='ntc_mlp.0'),
                nn.ReLU(inplace=True),
                LinearWithCache(self.feat_dim, 3*self.n_offsets+self.feat_dim, ref_nn, nn_prefix='ntc_mlp.2'),
            ).cuda()

            self.ntc_anc_mlp = nn.Sequential(
                LinearWithCache(self.latent_dim, self.feat_dim, ref_nn, nn_prefix='ntc_anc_mlp.0'),
                nn.ReLU(True),
                LinearWithCache(self.feat_dim, 3+4, ref_nn, nn_prefix='ntc_anc_mlp.2'),
            ).cuda()
        else:
            self.ntc_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, 3*self.n_offsets+self.feat_dim),
            ).cuda()

            self.ntc_anc_mlp = nn.Sequential(
                nn.Linear(self.latent_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3+4),
            ).cuda()

        self.qb = qb

    def encode_net(self, file_name):
        params_symbols = {}
        params_symbols["ntc_mlp"], bits_ntc_mlp = encode_mlp(self.ntc_mlp, self.qb, os.path.join(file_name, 'ntc_mlp'))
        params_symbols["ntc_anc_mlp"], bits_ntc_anc_mlp = encode_mlp(self.ntc_anc_mlp, self.qb, os.path.join(file_name, 'ntc_anc_mlp'))
        total_bits = bits_ntc_mlp + bits_ntc_anc_mlp
        return params_symbols, total_bits

    def decode_net(self, file_name):
        params_symbols = {}
        params_symbols["ntc_mlp"] = decode_mlp(self.ntc_mlp, self.qb, os.path.join(file_name, 'ntc_mlp'))
        params_symbols["ntc_anc_mlp"] = decode_mlp(self.ntc_anc_mlp, self.qb, os.path.join(file_name, 'ntc_anc_mlp'))

    def estimate_params_bits(self):
        estimated_bits = estimate_mlp_bits(self.ntc_mlp, self.qb) + estimate_mlp_bits(self.ntc_anc_mlp, self.qb)
        return estimated_bits

    def forward(self, latent):
        if self.qb is not None:
            self.ntc_mlp = Quantize_MLP_ours.apply(self.ntc_mlp, self.qb)
            self.ntc_anc_mlp = Quantize_MLP_ours.apply(self.ntc_anc_mlp, self.qb)
        prm_emb = self.ntc_mlp(latent)
        d_offsets, d_feat = torch.split(prm_emb, [3*self.n_offsets, self.feat_dim], dim=1)
        anc_emb = self.ntc_anc_mlp(latent)
        d_anchor, anchor_rot = torch.split(anc_emb, [3, 4], dim=1)
        d_anchor = self.d_anchor_factor*d_anchor
        anchor_rot = torch.nn.functional.normalize(self.anchor_rot_factor * anchor_rot + torch.tensor([[1,0,0,0]],device="cuda"))
        anchor_rot = quaternion_to_rotation_matrix(anchor_rot)
        d_offsets = d_offsets.reshape(-1, self.n_offsets, 3)

        return d_feat, d_offsets, d_anchor, anchor_rot
