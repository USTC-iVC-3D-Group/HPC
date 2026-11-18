"""Gaussian likelihoods and the factorized latent entropy model."""
import torch
import torch.nn as nn
import numpy as np
from utils.encodings import use_clamp, encoder_gaussian, decoder_gaussian, write_floats, read_floats, write_uints, read_uints
import math
from torch.nn.parameter import Parameter
import os
import torchac

class Entropy_model_nnfree():

    def cal_bitrate(self, code, training):
        mean = torch.mean(code)
        std = torch.std(code) if code.numel() > 1 else code.new_zeros(())
        estimated_bits = torch.sum(self.get_bits(code, mean, std))
        return {"estimated_bits": estimated_bits, "mean":mean, "std":std}

    def get_bits(self, x, mu, sigma):

        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = GaussianLowerBound.apply(bits)
        return bits

    def compress(self, code, file_name):

        code = code.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(code).all():
            raise ValueError(f'Nonfinite network symbols: {file_name}')
        N_code = code.shape[0]
        mean = torch.mean(code)

        std = torch.std(code) if code.numel() > 1 else code.new_zeros(())
        std = torch.nan_to_num(std, nan=1e-5, posinf=1e10, neginf=1e-5).clamp(1e-5, 1e10)
        mean = torch.tensor(np.float32(mean.item()), dtype=torch.float32)
        std = torch.tensor(np.float32(std.item()), dtype=torch.float32)
        mean_tensor = mean*torch.ones_like(code)
        std_tensor = std*torch.ones_like(code)
        Q = torch.ones_like(code)
        bit, min_v, max_v = encoder_gaussian(code, mean_tensor, std_tensor, Q, file_name=file_name)

        assert file_name.endswith('.b')
        bit_header = 0
        header_b_name = f'{file_name[:-2]}_header.b'
        with open(header_b_name, 'wb') as f:
            bit_header += write_uints(f, (N_code,))
            bit_header += write_floats(f, (mean,))
            bit_header += write_floats(f, (std,))
            bit_header += write_floats(f, (min_v,))
            bit_header += write_floats(f, (max_v,))

        return bit + bit_header * 8

    def decompress(self, file_name):
        header_b_name = f'{file_name[:-2]}_header.b'
        with open(header_b_name, 'rb') as f:
            N_code = read_uints(f, 1)[0]
            mean = read_floats(f, 1)[0]
            std = read_floats(f, 1)[0]
            min_v = read_floats(f, 1)[0]
            max_v = read_floats(f, 1)[0]
        mean_tensor = mean*torch.ones([N_code])
        std_tensor = std*torch.ones([N_code])
        min_tensor = torch.tensor(min_v)
        max_tensor = torch.tensor(max_v)
        Q = torch.ones([N_code])
        code = decoder_gaussian(mean_tensor, std_tensor, Q, file_name, min_tensor, max_tensor)
        return code

class Entropy_gaussian(nn.Module):
    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q
    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q
        if use_clamp:
            if x_mean is None:
                x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        scale = torch.clamp(scale, min=1e-9)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5*Q)
        upper = m1.cdf(x + 0.5*Q)
        likelihood = torch.abs(upper - lower)
        likelihood = GaussianLowerBound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits

class GaussianLowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = np.logical_or(
            x.cpu().numpy() >= 1e-6, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if+0.0).cuda()
        return grad1 * t

class FactorizedLowerBound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-9)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        try:
            grad1[x<1e-9] = 0
        except RuntimeError:
            print("ERROR! grad1[x<1e-9] = 0")
            grad1 = g.clone()
        pass_through_if = np.logical_or(x.cpu().detach().numpy() >= 1e-9, g.cpu().detach().numpy()<0.0)
        t = torch.Tensor(pass_through_if+0.0).to(grad1.device)

        return grad1*t

class EntropyBottleneck(nn.Module):
    """The layer implements a flexible probability density model to estimate
    entropy of its input tensor, which is described in this paper:
    >"Variational image compression with a scale hyperprior"
    > J. Balle, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
    > https://arxiv.org/abs/1802.01436"""

    def __init__(self, channels, init_scale=8, filters=(3,3,3), qb=None):
        """create parameters.
        """
        super(EntropyBottleneck, self).__init__()
        self._likelihood_bound = 1e-9
        self._init_scale = float(init_scale)
        self._filters = tuple(int(f) for f in filters)
        self._channels = channels

        filters = (1,) + self._filters + (1,)
        scale = self._init_scale ** (1 / (len(self._filters) + 1))

        self._matrices = nn.ParameterList([])
        self._biases = nn.ParameterList([])
        self._factors = nn.ParameterList([])

        for i in range(len(self._filters) + 1):

            self.matrix = Parameter(torch.FloatTensor(channels, filters[i + 1], filters[i]))
            init_matrix = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix.data.fill_(init_matrix)
            self.matrix.data = self.matrix.data + torch.randn(self.matrix.data.shape, device=self.matrix.data.device)*1e-5
            self._matrices.append(self.matrix)

            self.bias = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            init_bias = torch.FloatTensor(np.random.uniform(-0.5, 0.5, self.bias.size()))
            self.bias.data.copy_(init_bias)# copy or fill?
            self.bias.data = self.bias.data + torch.randn(self.bias.data.shape, device=self.bias.data.device)*1e-5
            self._biases.append(self.bias)

            self.factor = Parameter(torch.FloatTensor(channels, filters[i + 1], 1))
            self.factor.data.fill_(0.0)
            self.factor.data = self.factor.data + torch.randn(self.factor.data.shape, device=self.factor.data.device)*1e-5
            self._factors.append(self.factor)

        self.qb = qb

    def _logits_cumulative(self, inputs):
        """Evaluate logits of the cumulative densities.

        Arguments:
        inputs: The values at which to evaluate the cumulative densities,
            expected to have shape `(channels, 1, batch)`.

        Returns:
        A tensor of the same shape as inputs, containing the logits of the
        cumulatice densities evaluated at the the given inputs.
        """
        from utils.networks import Quantize_Parameters

        logits = inputs
        for i in range(len(self._filters) + 1):
            if self.qb is not None and not getattr(self, "codec_frozen", False):
                self._matrices[i].data = Quantize_Parameters(self._matrices[i].data, self.qb)
                self._biases[i].data = Quantize_Parameters(self._biases[i].data, self.qb)
                self._factors[i].data = Quantize_Parameters(self._factors[i].data, self.qb)
            matrix = torch.nn.functional.softplus(self._matrices[i])
            logits = torch.matmul(matrix, logits)
            logits += self._biases[i]
            factor = torch.tanh(self._factors[i])
            logits += factor * torch.tanh(logits)

        return logits

    def _likelihood(self, inputs):
        """Estimate the likelihood.
        inputs shape: [points, channels]
        """

        inputs = inputs.permute(1, 0).contiguous()# [channels, points]
        shape = inputs.size()# [channels, points]
        inputs = inputs.view(shape[0], 1, -1)# [channels, 1, points]
        inputs = inputs.to(self.matrix.device)

        lower = self._logits_cumulative(inputs - 0.5)
        upper = self._logits_cumulative(inputs + 0.5)
        sign = -torch.sign(torch.add(lower, upper)).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        likelihood = likelihood.view(shape)
        likelihood = likelihood.permute(1, 0)

        return likelihood

    def encode_net(self, file_name):
        from utils.networks import encode_tensor

        symbols_dict = {"m": [], "b": [], "f":[]}
        bits = 0
        for i in range(len(self._filters) + 1):
            if self.qb is not None:
                m_dict, m_bits = encode_tensor(self._matrices[i].data, self.qb, os.path.join(file_name, f'{str(i)}_m.b'))
                b_dict, b_bits = encode_tensor(self._biases[i].data, self.qb, os.path.join(file_name, f'{str(i)}_b.b'))
                f_dict, f_bits = encode_tensor(self._factors[i].data, self.qb, os.path.join(file_name, f'{str(i)}_f.b'))
                symbols_dict["m"].append(m_dict)
                symbols_dict["b"].append(b_dict)
                symbols_dict["f"].append(f_dict)
                bits = bits + m_bits + b_bits + f_bits
        return symbols_dict, bits

    def decode_net(self, file_name):
        from utils.networks import decode_tensor

        symbols_dict = {"m": [], "b": [], "f":[]}
        for i in range(len(self._filters) + 1):
            if self.qb is not None:
                m_dict = decode_tensor(os.path.join(file_name, f'{str(i)}_m.b'))
                b_dict = decode_tensor(os.path.join(file_name, f'{str(i)}_b.b'))
                f_dict = decode_tensor(os.path.join(file_name, f'{str(i)}_f.b'))
                symbols_dict["m"].append(m_dict)
                symbols_dict["b"].append(b_dict)
                symbols_dict["f"].append(f_dict)
        self.load_params_symbols(symbols_dict)

    def load_params_symbols(self, symbols_dict):
        from utils.networks import load_tensor_symbols_dict

        for i in range(len(self._filters) + 1):
            self._matrices[i].data.copy_(
                load_tensor_symbols_dict(symbols_dict["m"][i]).to(self._matrices[i].device).view_as(self._matrices[i]))
            self._biases[i].data.copy_(
                load_tensor_symbols_dict(symbols_dict["b"][i]).to(self._biases[i].device).view_as(self._biases[i]))
            self._factors[i].data.copy_(
                load_tensor_symbols_dict(symbols_dict["f"][i]).to(self._factors[i].device).view_as(self._factors[i]))

    def forward(self, inputs):
        """Pass a tensor through the bottleneck.
        """

        outputs = inputs
        likelihood = self._likelihood(outputs)

        likelihood = FactorizedLowerBound.apply(likelihood)

        return outputs, likelihood

    def _pmf_to_cdf(self, pmf):
        cdf = pmf.cumsum(dim=-1)
        spatial_dimensions = pmf.shape[:-1] + (1,)
        zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
        cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
        cdf_with_0 = cdf_with_0.clamp(max=1.)

        return cdf_with_0

    @torch.no_grad()
    def codec_cdf(self, min_v, max_v, channels):
        symbols = torch.arange(float(min_v), float(max_v) + 1)
        symbols = symbols.reshape(-1, 1).repeat(1, channels)
        pmf = self._likelihood(symbols).clamp(min=self._likelihood_bound)
        cdf = self._pmf_to_cdf(pmf.permute(1, 0)).detach().cpu()
        return cdf

    @torch.no_grad()
    def compress(self, inputs):

        values = inputs

        min_v = values.min().detach().float()
        max_v = values.max().detach().float()

        values_norm = values - min_v

        values_norm = values_norm.to(torch.int16)

        cdf = self.codec_cdf(min_v, max_v, values.shape[-1])

        out_cdf = cdf.unsqueeze(0).repeat(values_norm.shape[0], 1, 1).detach().cpu()
        strings = torchac.encode_float_cdf(out_cdf, values_norm.cpu(), check_input_bounds=True)

        return strings, min_v, max_v

    @torch.no_grad()
    def decompress(self, strings, min_v, max_v, shape, channels):

        cdf = self.codec_cdf(min_v, max_v, channels)

        out_cdf = cdf.unsqueeze(0).repeat(shape[0], 1, 1).detach().cpu()
        values = torchac.decode_float_cdf(out_cdf, strings)
        values = values.float()
        values += min_v

        return values
