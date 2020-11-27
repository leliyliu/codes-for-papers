"""
    Private:
@article{zhang2019adaptive,
  title={Adaptive Precision Training: Quantify Back Propagation in Neural Networks with Fixed-point Numbers},
  author={Zhang, Xishan and Liu, Shaoli and Zhang, Rui and Liu, Chang and Huang, Di and Zhou, Shiyi and Guo, Jiaming and Kang, Yu and Guo, Qi and Du, Zidong and others},
  journal={arXiv preprint arXiv:1911.00361},
  year={2019}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

__all__ = ['LinearAQ', 'Conv2dAQ']


class cfg(object):
    # a config file for quantize

    # 输入量化初始位宽
    input_bits = 8
    # 权重量化初始位宽
    weight_bits = 8
    # 梯度量化初始位宽
    grad_bits = 8

    # 是否自适应周期
    input_dynamic_interval = True
    weight_dynamic_interval = True
    grad_dynamic_interval = True
    # 是否自适应位宽
    input_dynamic_n = True
    weight_dynamic_n = True
    grad_dynamic_n = True

    # 非自适应周期时更新间隔
    input_interval = 1
    weight_interval = 1
    grad_interval = 1

    # 更新阈值
    update_bitnum_th = 0.03

    # 每次增大比特数
    add_bit = 8

    # 最大比特位宽
    max_bit = 24

    # 每个Epoch代数
    epoch_step = 313

    # 更新参数
    alpha = 0.04
    beta = 0.1
    gamma = 2
    delta = 100


class LinearAQ(nn.Linear):
    """quantized Linear."""

    def __init__(self, in_features, out_features, bias=True,
                 input_bits=cfg.input_bits, weight_bits=cfg.weight_bits,
                 grad_bits=cfg.grad_bits):
        super(LinearAQ, self).__init__(in_features, out_features, bias)

        self.register_buffer('cur_step', torch.tensor([0], dtype=torch.float))
        # the parameter is: bits, offset, shift, scale
        # No want to write a lots of variables ;) Function Concept
        self.register_buffer('input_param', torch.tensor(
            [input_bits, 0, 0, 100], dtype=torch.float))
        self.register_buffer('weight_param', torch.tensor(
            [weight_bits, 0, 0, 100], dtype=torch.float))
        self.register_buffer('grad_param', torch.tensor(
            [grad_bits, 0, 0, 100], dtype=torch.float))

    def forward(self, input):
        q_input = quantize(input,
                           param=self.input_param,
                           cur_step=self.cur_step,
                           interval=cfg.input_interval,
                           dynamic_n=cfg.input_dynamic_n,
                           dynamic_interval=cfg.input_dynamic_interval)
        q_weight = quantize(self.weight,
                            param=self.weight_param,
                            cur_step=self.cur_step,
                            interval=cfg.weight_interval,
                            dynamic_n=cfg.weight_dynamic_n,
                            dynamic_interval=cfg.weight_dynamic_interval)
        q_output = F.linear(q_input, q_weight, self.bias)
        if self.grad_param[0] > 0:
            q_output = quantize_grad(q_output,
                                     param=self.grad_param,
                                     cur_step=self.cur_step,
                                     interval=cfg.grad_interval,
                                     dynamic_n=cfg.grad_dynamic_n,
                                     dynamic_interval=cfg.grad_dynamic_interval)
        #:print(self.cur_step, self.shift_param)

        if self.training:
            self.cur_step[0] = self.cur_step[0] + 1

        return q_output


# class Conv1d(nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True,
#                  input_bits=cfg.input_bits, weight_bits=cfg.weight_bits,
#                  grad_bits=cfg.grad_bits):
#         super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
#                                      stride, padding, dilation, groups, bias)
#
#         self.register_buffer('cur_step', torch.tensor(
#             [0], dtype=torch.float))
#         # the parameter is: bits, offset, shift, scale
#         # No want to write a lots of variables ;) Function Concept
#         self.register_buffer('input_param', torch.tensor(
#             [input_bits, 0, 0, 100], dtype=torch.float))
#         self.register_buffer('weight_param', torch.tensor(
#             [weight_bits, 0, 0, 100], dtype=torch.float))
#         self.register_buffer('grad_param', torch.tensor(
#             [grad_bits, 0, 0, 100], dtype=torch.float))
#
#     def forward(self, input):
#         q_input = QB.quantize(input,
#                               param=self.input_param,
#                               cur_step=self.cur_step,
#                               interval=cfg.input_interval,
#                               dynamic_n=cfg.input_dynamic_n,
#                               dynamic_interval=cfg.input_dynamic_interval)
#         q_weight = QB.quantize(self.weight,
#                                param=self.weight_param,
#                                cur_step=self.cur_step,
#                                interval=cfg.weight_interval,
#                                dynamic_n=cfg.weight_dynamic_n,
#                                dynamic_interval=cfg.weight_dynamic_interval)
#
#         q_output = F.conv1d(q_input, q_weight, self.bias, self.stride,
#                             self.padding, self.dilation, self.groups)
#
#         if self.grad_param[0] > 0:
#             q_output = QB.quantize_grad(q_output,
#                                         param=self.grad_param,
#                                         cur_step=self.cur_step,
#                                         interval=cfg.grad_interval,
#                                         dynamic_n=cfg.grad_dynamic_n,
#                                         dynamic_interval=cfg.grad_dynamic_interval)
#
#         if self.training:
#             self.cur_step[0] = self.cur_step[0] + 1
#
#         return q_output


class Conv2dAQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 input_bits=cfg.input_bits, weight_bits=cfg.weight_bits,
                 grad_bits=cfg.grad_bits):  # , update_step=cfg.update_step):
        super(Conv2dAQ, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)
        self.register_buffer('cur_step', torch.tensor(
            [0], dtype=torch.float))
        # the parameter is: bits, offset, shift, scale
        # No want to write a lots of variables ;) Function Concept
        self.register_buffer('input_param', torch.tensor(
            [input_bits, 0, 0, 100], dtype=torch.float))
        self.register_buffer('weight_param', torch.tensor(
            [weight_bits, 0, 0, 100], dtype=torch.float))
        self.register_buffer('grad_param', torch.tensor(
            [grad_bits, 0, 0, 100], dtype=torch.float))

    def forward(self, input):
        q_input = quantize(input,
                           param=self.input_param,
                           cur_step=self.cur_step,
                           interval=cfg.input_interval,
                           dynamic_n=cfg.input_dynamic_n,
                           dynamic_interval=cfg.input_dynamic_interval)
        q_weight = quantize(self.weight,
                            param=self.weight_param,
                            cur_step=self.cur_step,
                            interval=cfg.weight_interval,
                            dynamic_n=cfg.weight_dynamic_n,
                            dynamic_interval=cfg.weight_dynamic_interval)

        q_output = F.conv2d(q_input, q_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if self.grad_param[0] > 0:
            q_output = quantize_grad(q_output,
                                     param=self.grad_param,
                                     cur_step=self.cur_step,
                                     interval=cfg.grad_interval,
                                     dynamic_n=cfg.grad_dynamic_n,
                                     dynamic_interval=cfg.grad_dynamic_interval)

        if self.training:
            self.cur_step[0] = self.cur_step[0] + 1

        return q_output


def get_update_step(shift, mv, cur_step, diff_bit, dynamic_interval, interval=1):
    if cur_step < cfg.epoch_step:
        if dynamic_interval is True:
            if cur_step < (cfg.epoch_step / 100):
                next_update_step = cur_step + 1
            else:
                diff1 = cfg.alpha * torch.abs(shift - mv)
                diff2 = cfg.delta * diff_bit ** 2
                diff = torch.max(diff1, diff2)

                I = torch.max(torch.round(cfg.beta / diff - cfg.gamma),
                              torch.tensor(1, dtype=torch.float).cuda())
                if I > cfg.epoch_step:
                    I = cfg.epoch_step
                next_update_step = cur_step + I
        else:
            next_update_step = cur_step + interval
    else:
        next_update_step = cur_step + cfg.epoch_step

    mv = cfg.alpha * shift + (1 - cfg.alpha) * mv
    return mv, next_update_step


def get_new_shift(data, bitnum):
    Z = data.abs().max()
    if Z > 0:
        shift = torch.ceil(torch.log2(Z / (2 ** (bitnum - 1) - 1)))
    else:
        shift = torch.tensor(-12.0).cuda()
    return shift


def float2fix(data, bitnum, shift):  # TODO: no clamp ??
    output = data.clone()
    step = 2 ** shift
    output.div_(step).round_().clamp(-2 ** bitnum, 2 ** bitnum - 1).mul_(step)
    return output


class Quantize(Function):
    @classmethod
    def forward(cls, ctx, input, param=None, cur_step=None, interval=1, dynamic_n=False, dynamic_interval=False):
        bitnum, shift, update_step, mv = param[:]

        # print(cur_step, bitnum, shift)
        if cur_step == update_step:
            while True:
                shift = get_new_shift(input, bitnum)
                output = float2fix(input, bitnum, shift)
                input_mean = torch.mean(torch.abs(input))
                output_mean = torch.mean(torch.abs(output))
                diff_bit = torch.log2(torch.abs((input_mean - output_mean) / input_mean) + 1)

                if dynamic_n is False:
                    break
                else:
                    if diff_bit > cfg.update_bitnum_th and bitnum < cfg.max_bit:
                        bitnum = bitnum + cfg.add_bit
                        mv = mv - cfg.add_bit
                    else:
                        break
            mv, update_step = get_update_step(shift, mv, cur_step, diff_bit, dynamic_interval, interval)
        else:
            output = float2fix(input, bitnum, shift)
            mv = cfg.alpha * shift + (1 - cfg.alpha) * mv

        param[:] = torch.tensor([bitnum, shift, update_step, mv], dtype=torch.float).cuda()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None


class QuantizeGrad(Function):

    @classmethod
    def forward(cls, ctx, input, param=None, cur_step=None, interval=1, dynamic_n=False, dynamic_interval=False):
        ctx.param = param
        ctx.cur_step = cur_step
        ctx.interval = interval
        ctx.dynamic_n = dynamic_n
        ctx.dynamic_interval = dynamic_interval
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = Quantize().apply(grad_output, ctx.param,
                                      ctx.cur_step - 1, ctx.interval, ctx.dynamic_n, ctx.dynamic_interval)
        # print(ctx.cur_step, ctx.param)
        # grad_input = grad_output * 0.0
        #        print(ctx.cur_step, grad_input.mean(), grad_output.mean())
        return grad_input, None, None, None, None, None


def quantize(x, param=None, cur_step=None, interval=1, dynamic_n=False, dynamic_interval=False):
    return Quantize().apply(x, param, cur_step, interval, dynamic_n, dynamic_interval)


def quantize_grad(x, param=None, cur_step=None, interval=1, dynamic_n=False, dynamic_interval=False):
    return QuantizeGrad().apply(x, param, cur_step, interval, dynamic_n, dynamic_interval)
