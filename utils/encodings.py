import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import torchac
import struct

try:
    import _gridencoder as _backend
except ImportError:
    _backend = None  # CPU entropy-codec tests do not require the CUDA grid.
import os

anchor_round_digits = 16
Q_anchor = 1/(2 ** anchor_round_digits - 1)
use_clamp = True

def get_binary_vxl_size(binary_vxl):

    ttl_num = binary_vxl.numel()

    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = pos_num / ttl_num  #  + 1e-6
    Pg = torch.clamp(Pg, min=1e-6, max=1-1e-6)
    pos_prob = Pg
    neg_prob = (1 - Pg)
    pos_bit = pos_num * (-torch.log2(pos_prob))
    neg_bit = neg_num * (-torch.log2(neg_prob))
    ttl_bit = pos_bit + neg_bit
    ttl_bit += 32  # Pg

    return Pg, ttl_bit, ttl_bit.item()/8.0/1024/1024, ttl_num

def encoder_gaussian(x, mean, scale, Q, file_name='tmp.b'):
    dir_name = os.path.dirname(file_name)
    if dir_name != '' and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert x.shape == mean.shape == scale.shape == Q.shape
    x_int_round = torch.round(x / Q)  # [100]
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(x.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]

    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    del samples
    del mean
    del scale
    del GD
    x_int_round_idx = (x_int_round - min_value).to(torch.int16)
    assert (x_int_round_idx.to(torch.int32) == x_int_round - min_value).all()

    byte_stream = torchac.encode_float_cdf(lower.cpu(), x_int_round_idx.cpu(), check_input_bounds=True)
    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    bit_len = len(byte_stream)*8
    torch.cuda.empty_cache()
    return bit_len, min_value, max_value

def decoder_gaussian(mean, scale, Q, file_name='tmp.b', min_value=-100, max_value=100):
    assert file_name.endswith('.b')
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=mean.dtype, device=mean.device).repeat(mean.shape[0])
    assert mean.shape == scale.shape == Q.shape
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1 + 1)).to(
        torch.float).to(mean.device)  # from min_value to max_value+1. shape = [max_value+1+1 - min_value]
    samples = samples.unsqueeze(0).repeat(mean.shape[0], 1)  # [100, max_value+1+1 - min_value]
    mean = mean.unsqueeze(-1).repeat(1, samples.shape[-1])
    scale = scale.unsqueeze(-1).repeat(1, samples.shape[-1])
    GD = torch.distributions.normal.Normal(mean, scale)
    lower = GD.cdf((samples - 0.5) * Q.unsqueeze(-1))
    with open(file_name, 'rb') as fin:
        byte_stream_d = fin.read()
    sym_out = torchac.decode_float_cdf(lower.cpu(), byte_stream_d).to(mean.device).to(torch.float32)
    x = sym_out + min_value
    x = x * Q
    torch.cuda.empty_cache()
    return x

def encoder_factorized(x, factorized_model, file_name):
    byte_stream, min_v, max_v = factorized_model.compress(x.cpu())
    bit_len = len(byte_stream) * 8

    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len, min_v, max_v

def decoder_factorized(file_name, factorized_model, min_v, max_v, shape):
    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()
    channels = shape[-1]
    decode_v = factorized_model.decompress(byte_stream, min_v, max_v, shape, channels)
    return decode_v

def encoder(x, p, file_name):
    x = x.detach().cpu()
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)

    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(((x+1)/2)).to(torch.int16)
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)

    bit_len = len(byte_stream) * 8

    with open(file_name, 'wb') as fout:
        fout.write(byte_stream)
    return bit_len

def decoder(p, file_name):
    dvc = p.device
    p = p.detach().cpu()
    assert file_name[-2:] == '.b'
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)

    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)

    with open(file_name, 'rb') as fin:
        byte_stream = fin.read()

    sym_out = torchac.decode_float_cdf(output_cdf, byte_stream)
    sym_out = (sym_out * 2 - 1).to(torch.float32)
    return sym_out.to(dvc)

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)

        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask

class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            input_min = input_mean - 15_000 * Q
            input_max = input_mean + 15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class Quantize_anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchors, min_v, max_v):

        interval = ((max_v - min_v) * Q_anchor + 1e-6)  # avoid 0, if max_v == min_v

        quantized_v = torch.div(anchors - min_v, interval, rounding_mode='floor')
        quantized_v = torch.clamp(quantized_v, 0, 2 ** anchor_round_digits - 1)
        anchors_q = quantized_v * interval + min_v
        return anchors_q, quantized_v
    @staticmethod
    def backward(ctx, grad_output, tmp):  # tmp is for quantized_v:)
        return grad_output, None, None

class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, min_level_id=None, n_levels_calc=1, binary_vxl=None, PV=0):

        inputs = inputs.contiguous()

        Rb = 128
        if binary_vxl is not None:
            binary_vxl = binary_vxl.contiguous()
            Rb = binary_vxl.shape[-1]
            assert len(binary_vxl.shape) == inputs.shape[-1]

        N, num_dim = inputs.shape # batch size, coord dim # N_rays, 3
        n_levels = offsets_list.shape[0] - 1 # level # 层数=16
        n_features = embeddings.shape[1] # embedding dim for each level # 就是channel数=2

        max_level_id = min_level_id + n_levels_calc

        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings.to(torch.half)

        outputs = torch.empty(n_levels_calc, N, n_features, device=inputs.device, dtype=embeddings.dtype)  # 创建一个buffer给cuda填充

        if calc_grad_inputs:  # inputs.requires_grad
            dy_dx = torch.empty(N, n_levels_calc * num_dim * n_features, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_forward(
                inputs,
                embeddings,

                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_forward(
                inputs,
                embeddings,

                offsets_list,
                resolutions_list,
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx,
                binary_vxl,
                min_level_id
                )

        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels_calc * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl)
        ctx.dims = [N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV]  # min_level_id是否要单独save为tensor

        return outputs

    @staticmethod

    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl = ctx.saved_tensors
        N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV = ctx.dims

        grad = grad.view(N, n_levels_calc, n_features).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,

                offsets_list[min_level_id:max_level_id+1],
                resolutions_list[min_level_id:max_level_id],
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                None
                )
        else:
            _backend.grid_encode_backward(
                grad,
                inputs,
                embeddings,

                offsets_list,
                resolutions_list,
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx,
                grad_inputs,
                binary_vxl,
                min_level_id
                )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None
grid_encode = _grid_encode.apply
class GridEncoder(nn.Module):
    def __init__(self,
                 num_dim=3,
                 n_features=2,
                 resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736),
                 log2_hashmap_size=19,
                 ste_binary = True,
                 ste_multistep = False,
                 add_noise = False,
                 Q = 1
                 ):
        super().__init__()

        resolutions_list = torch.tensor(resolutions_list).to(torch.int)
        n_levels = resolutions_list.numel()

        self.num_dim = num_dim # coord dims, 2 or 3
        self.n_levels = n_levels # num levels, each level multiply resolution by 2
        self.n_features = n_features # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q

        offsets_list = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim)  # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8)  # make divisible
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)

        self.n_params = offsets_list[-1] * n_features

        self.params = nn.Parameter(torch.empty(offset, n_features))

        self.reset_parameters()

        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 1e-4
        self.params.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: num_dim={self.num_dim} n_levels={self.n_levels} n_features={self.n_features} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.n_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.params.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, min_level_id=None, max_level_id=None, test_phase=False, outspace_params=None, binary_vxl=None, PV=0):

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        if outspace_params is not None:
            params = nn.Parameter(outspace_params)
        else:
            params = self.params

        if self.ste_binary:
            embeddings = STE_binary.apply(params)

        elif (self.add_noise and not test_phase):
            embeddings = params + (torch.rand_like(params) - 0.5) * (1 / self.Q)
        elif (self.ste_multistep) or (self.add_noise and test_phase):
            embeddings = STE_multistep.apply(params, self.Q)
        else:
            embeddings = params

        min_level_id = 0 if min_level_id is None else max(min_level_id, 0)
        max_level_id = self.n_levels if max_level_id is None else min(max_level_id, self.n_levels)
        n_levels_calc = max_level_id - min_level_id

        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, inputs.requires_grad, min_level_id, n_levels_calc, binary_vxl, PV)
        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        return outputs

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_floats(fd, values, fmt=">{:d}f"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def read_floats(fd, n, fmt=">{:d}f"):
    sz = struct.calcsize("f")
    return struct.unpack(fmt.format(n), fd.read(n * sz))
