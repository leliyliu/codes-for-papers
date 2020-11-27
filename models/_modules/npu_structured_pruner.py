"""
Structured pruning for custom NPU.
&
Qcode quantization.

weight.shape = (out_channel, in_channel, kernel_size, kernel_size)

one group: weight[ m*32 : (m+1)32, n*32 : (n+1) * 32, i, j].sum(axis=1).max()

worst case decides the number of processing cycle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['Conv2dNPU']


def get_default_kwargs_q(kwargs_q):
    default = {
        'non_zero_num': 32,
        'pe_size': 32,
        'total_iter': 0,  # incremental sparse iteration
        'beta': 0.5,
        'INS': False,
    }
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    assert 7 <= kwargs_q['non_zero_num'] <= 32, 'the non_zero_num must be greater than 6 and less than 33 !!'
    return kwargs_q


def get_npu_structured_sparsity_mask(param, non_zero_num: int, pe_size=32):
    if non_zero_num >= 32:
        return torch.ones_like(param)
    (out_channel, in_channel, k, k) = param.shape
    part = math.ceil(in_channel / pe_size)
    param_reshape = param.transpose(0, 1).reshape(in_channel, -1)  # in, out * k * k
    mask = torch.zeros_like(param_reshape)  # in, out * k * k
    for i in range(part):
        param_reshape_part = param_reshape[i * pe_size: (i + 1) * pe_size, :]  # in, out * k * k
        bottomk, _ = torch.topk(param_reshape_part.abs().transpose(0, 1), non_zero_num + 1, largest=True, sorted=True)
        threshold = bottomk.data[:, -1]
        mask_part = torch.gt(torch.abs(param_reshape_part), threshold).type(param.type())
        mask[i * pe_size: (i + 1) * pe_size, :] = mask_part
    return mask.reshape(in_channel, out_channel, k, k).transpose(0, 1)  # out, in, k, k


class Conv2dNPU(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(Conv2dNPU, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.register_buffer('init_state', torch.zeros(1))  # sparsity
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.iter = 0
        self.total_iter = self.kwargs_q['total_iter']
        self.beta = self.kwargs_q['beta']
        self.INS = self.kwargs_q['INS']
        self.ins_iter = self.total_iter * self.beta
        if self.INS:
            self.register_buffer('non_zero_num_ins', torch.zeros(1).fill_(15))

    def forward(self, x):
        # 1. pruning weights
        if self.INS:
            if self.init_state[0] == 0 and self.training:  # lazy fix+incremental mask for pruning
                if self.iter % int(self.ins_iter / 10) == 0 and self.non_zero_num_ins >= self.kwargs_q['non_zero_num']:
                    print('{} init mask {}/32'.format(self._get_name(), int(self.non_zero_num_ins.item())))
                    self.mask.copy_(
                        get_npu_structured_sparsity_mask(self.weight, int(self.non_zero_num_ins.item()),
                                                         pe_size=self.kwargs_q['pe_size']))
                    self.weight.data.mul_(self.mask)
                    if self.non_zero_num_ins == self.kwargs_q['non_zero_num']:
                        self.init_state[0] += 1
                        print('End of the ins phase. non_zero_num:{}'.format(self.non_zero_num_ins.item()))
                    else:
                        self.non_zero_num_ins -= 1
                self.iter += 1
            elif self.init_state[0] == 0 and self.iter == 0 and not self.training:  # post-training sparsity
                raise NotImplementedError('Please set INS = False')
        elif self.init_state[0] == 0:
            # todo: npu_structured_sparsity_mask
            print('{} init mask {}/32'.format(self._get_name(), self.kwargs_q['non_zero_num']))
            self.mask.copy_(
                get_npu_structured_sparsity_mask(self.weight, self.kwargs_q['non_zero_num'],
                                                 pe_size=self.kwargs_q['pe_size']))
            self.weight.data.mul_(self.mask)
            self.init_state[0] += 1

        w_s = self.weight * self.mask
        return F.conv2d(x, w_s, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dNPU, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs_q)
