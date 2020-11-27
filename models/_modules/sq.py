import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models._modules import *

__all__ = ['Conv2dSQ']


def get_default_kwargs_q(kwargs_q):
    default = {
        'sparsity': 0.0,
        'nbits_a': 4,
        'nbits_w': 4,
        'total_iter': 0,  # incremental sparse iteration
        'beta': 0.5,
        'INS': False,
    }
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    assert 0.0 <= kwargs_q['sparsity'] < 1.0, 'the sparsity must be greater than 0 and less than 1 !!'
    return kwargs_q


def sin_ins(iter, total_iter, s_exp, s_init=0.3, beta=0.5, INS=True):
    if not INS:
        return s_exp
    if iter >= beta * total_iter:
        return s_exp
    return (s_exp - s_init) * math.sin(math.pi * iter / (2 * beta * total_iter)) + s_init


class Conv2dSQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(Conv2dSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits_a = self.kwargs_q['nbits_a']
        self.nbits_w = self.kwargs_q['nbits_w']
        self.sparsity = self.kwargs_q['sparsity']
        self.total_iter = self.kwargs_q['total_iter']
        self.beta = self.kwargs_q['beta']
        self.INS = self.kwargs_q['INS']
        self.iter = 0
        self.get_sparsity = partial(sin_ins, total_iter=self.total_iter, s_exp=self.sparsity, beta=self.beta,
                                    INS=self.INS)
        self.ins_iter = self.total_iter * self.beta
        if self.nbits_a <= 0:
            self.register_parameter('scale_a', None)
        else:
            self.scale_a = Parameter(torch.Tensor(1))
        if self.nbits_w <= 0:
            self.register_parameter('scale_w', None)
        else:
            self.scale_w = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(3))  # [sparsity, qa, qw]
        if self.sparsity <= 1e-5:
            self.register_buffer('mask', None)
        else:
            self.register_buffer('mask', torch.ones_like(self.weight))

    def forward(self, x):
        # 1. pruning weights
        if self.sparsity > 1e-5:
            if self.INS:
                if self.init_state[0] == 0 and self.training:  # lazy fix+incremental mask for pruning
                    if self.iter % int(self.ins_iter / 10) == 0 or self.iter == self.ins_iter:
                        # self.weight.data.mul_(self.mask)
                        # self.old_mask.copy_(self.mask)  # debug
                        self.mask.copy_(get_sparsity_mask(self.weight, self.get_sparsity(self.iter)))
                        # if (self.old_mask - self.mask).min() < 0:  # debug
                        #     ipdb.set_trace()
                        #     print('dynamic!!!')
                        self.weight.data.mul_(self.mask)
                    if self.iter >= self.ins_iter:
                        self.init_state[0] += 1
                    self.iter += 1
                elif self.init_state[0] == 0 and self.iter == 0 and not self.training:  # post-training sparsity
                    # Please set INS = False
                    raise NotImplementedError('Please set INS = False')
                    # self.mask.copy_(get_sparsity_mask(self.weight, self.sparsity))
                    # self.weight.data.mul_(self.mask)
                    # self.init_state[0] += 1
            else:
                if self.init_state[0] == 0:  # post-training sparsity
                    self.mask.copy_(get_sparsity_mask(self.weight, self.sparsity))
                    self.weight.data.mul_(self.mask)
                    self.init_state[0] += 1

            w_s = self.weight * self.mask
        else:
            w_s = self.weight
        # 2. quantize activation
        #  for now.
        if self.nbits_a <= 0:
            x_q = x
        else:
            """asymmetric quantization
             Qn = 0
            Qp = 2 ** self.nbits_a - 1
            if self.init_state[1] == 0:
                # todo: remove outlier
                min_a = x.min()
                max_a = x.max()
                self.scale_a.fill_((max_a - min_a) / Qp)
                self.zero_point_a.fill_((min_a / self.scale_a).round().item())
                self.init_state[1] += 1
            x_q = (((x / self.scale_a).round() - self.zero_point_a).clamp(
                Qn, Qp) + self.zero_point_a) * self.scale_a
            """
            if x.min() > -1e-5:
                Qn = 0
                Qp = 2 ** self.nbits_a - 1
            else:
                Qn = -2 ** (self.nbits_a - 1)
                Qp = 2 ** (self.nbits_a - 1) - 1
            if self.init_state[1] == 0:
                if self.nbits_a < 8:
                    self.scale_a.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
                else:  # todo: initial value for post-training quantization; outlier value
                    self.scale_a.data.copy_(x.abs().max() / Qp)
                self.init_state[1] += 1
            g = 1.0 / math.sqrt(x.numel() * Qp)
            # Method1:
            scale_a = grad_scale(self.scale_a, g)
            x_q = round_pass((x / scale_a).clamp(Qn, Qp)) * scale_a

            # x_q = FunLSQ.apply(x, self.scale_a, g, Qn, Qp)
        # 3. quantize weight
        if self.nbits_w <= 0:
            w_q = w_s
        else:
            """ linear asymmetric quantization.
            Qn = 0
            Qp = 2 ** self.nbits_w - 1
            if self.init_state[2] == 0:
                min_w = w_s.min()
                max_w = w_s.max()
                self.scale_w.fill_((max_w - min_w) / Qp)
                self.zero_point_w.fill_((min_w / self.scale_w).round().item())
                self.init_state[2] += 1
            w_q = (((w_s / self.scale_w).round() - self.zero_point_w).clamp(
                Qn, Qp) + self.zero_point_w) * self.scale_w
            """

            Qn = -2 ** (self.nbits_w - 1)
            Qp = 2 ** (self.nbits_w - 1) - 1
            if self.init_state[2] == 0:
                if self.nbits_w < 8:
                    self.scale_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
                else:  # todo: initial value for post-training quantization
                    self.scale_w.data.copy_(self.weight.abs().max() / Qp)
                self.init_state[2] += 1
            g = 1.0 / math.sqrt(w_s.numel() * Qp)
            # Method1:
            scale_w = grad_scale(self.scale_w, g)
            w_q = round_pass((w_s / scale_w).clamp(Qn, Qp)) * scale_w
            # Method2:
            # w_q = FunLSQ.apply(w_s, self.scale_w, g, Qn, Qp)
        return F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dSQ, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs_q)
