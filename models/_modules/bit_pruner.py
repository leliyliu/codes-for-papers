import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from functools import partial
from models._modules._quan_base import linear_dequantize, linear_quantize_clamp, get_quantized_range

__all__ = ['Conv2dTruncation', 'LinearTruncation', 'ActTruncation',
           'Conv2dBPv1',
           'Conv2dBPv2', 'LinearBPv2']


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def get_default_kwargs_truncation(kwargs_q):
    default = {
        'nbits': 32,
    }
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


def get_default_kwargs_bp1(kwargs_q):
    default = {
        'nbits': 32,

        'expected_bit_sparsity': None,
        'increase_factor': 0.33,
    }
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


def get_default_kwargs_bp2(kwargs_q):
    default = {
        'nbits': 32,

        'log': False
    }
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class ActBPv2(nn.Module):
    def __init__(self, **kwargs):
        super(ActBPv2, self).__init__()
        self.kwargs = get_default_kwargs_bp2(kwargs)
        self.nbits = self.kwargs['nbits']
        self.log = self.kwargs['log']
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('init_state', None)
            self.register_buffer('radix_position', None)
        else:
            self.register_buffer('init_state', torch.zeros(1))
            self.register_buffer('radix_position', torch.zeros(1))

    def forward(self, x):
        if self.radix_position is None:
            return x
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0 and x.shape[0] >= 32:  # batch size >=0
            # batch size; remove sqrt[bs] outliers. topk
            batch_size = x.shape[0]
            il = torch.log2(x.abs().view(-1).topk(int(math.sqrt(batch_size)))[0][-1]) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            # quantize
            alpha = 2 ** self.radix_position
            x_int = round_pass((x * alpha).clamp(Qn, Qp))
            # w_q = w_int / alpha
            self.init_state.data.fill_(1)
        # STE for quantized activation.
        x_bp = FunctionBitPruningSTE.apply(x, self.radix_position, self.log)
        return x_bp

    def extra_repr(self):
        s_prefix = super(ActBPv2, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


class LinearBPv2(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(LinearBPv2, self).__init__(in_features, out_features, bias)
        self.kwargs = get_default_kwargs_bp2(kwargs)
        self.nbits = self.kwargs['nbits']
        self.log = self.kwargs['log']
        self.mask = None
        if 'nbits_a' in self.kwargs:
            self.nbits_a = self.kwargs['nbits_a']
        else:
            self.nbits_a = 0
        if 0 < self.nbits_a <= 16:
            self.act_bpv2 = ActBPv2(nbits=self.nbits_a, log=self.log)
        else:
            self.act_bpv2 = None
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('init_state', None)
            self.register_buffer('radix_position', None)
            self.register_buffer('weight_int', None)
        else:
            self.register_buffer('init_state', torch.zeros(1))
            self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
            self.register_buffer('radix_position', torch.zeros(1))

    def forward(self, x):
        if self.act_bpv2 is not None:
            x = self.act_bpv2(x)
        if self.radix_position is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:  # need to set radix position
            # set radix position
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            # quantize
            alpha = 2 ** self.radix_position
            w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
            # w_q = w_int / alpha
            self.mask = (w_int.abs() > 0).float()
            self.init_state.data.fill_(1)
        weight_mask = FunctionStopGradient.apply(self.weight, self.mask)
        # STE for quantized weight.
        weight_bp = FunctionBitPruningSTE.apply(weight_mask, self.radix_position, self.log)
        w_int = (weight_bp * 2 ** self.radix_position).round()
        self.weight_int.data.copy_(w_int)
        out = F.linear(x, weight_bp, self.bias)
        return out

    def extra_repr(self):
        s_prefix = super(LinearBPv2, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


class Conv2dBPv2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dBPv2, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        self.kwargs = get_default_kwargs_bp2(kwargs)
        self.nbits = self.kwargs['nbits']
        self.log = self.kwargs['log']
        self.mask = None
        if 'nbits_a' in self.kwargs:
            self.nbits_a = self.kwargs['nbits_a']
        else:
            self.nbits_a = 0
        if 0 < self.nbits_a <= 16:
            self.act_bpv2 = ActBPv2(nbits=self.nbits_a, log=self.log)
        else:
            self.act_bpv2 = None
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('init_state', None)
            self.register_buffer('radix_position', None)
            self.register_buffer('weight_int', None)
        else:
            self.register_buffer('init_state', torch.zeros(1))
            self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
            self.register_buffer('radix_position', torch.zeros(1))

    def forward(self, x):
        if self.act_bpv2 is not None:
            x = self.act_bpv2(x)
        if self.radix_position is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:  # need to set radix position
            # set radix position
            il = torch.log2(torch.max(self.weight.max(), self.weight.min().abs())) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            # quantize
            alpha = 2 ** self.radix_position
            w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
            # w_q = w_int / alpha
            self.mask = (w_int.abs() > 0).float()
            self.init_state.data.fill_(1)
        weight_mask = FunctionStopGradient.apply(self.weight, self.mask)
        # STE for quantized weight.
        weight_bp = FunctionBitPruningSTE.apply(weight_mask, self.radix_position, self.log)
        w_int = (weight_bp * 2 ** self.radix_position).round()
        self.weight_int.data.copy_(w_int)
        out = F.conv2d(x, weight_bp, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        s_prefix = super(Conv2dBPv2, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


class Conv2dBPv1(nn.Conv2d):
    """
        Conv2d with Bit Pruning
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(Conv2dBPv1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                         padding, dilation, groups, bias)
        self.kwargs = get_default_kwargs_bp1(kwargs)
        self.expected_bit_sparsity = self.kwargs['expected_bit_sparsity']
        self.nbits = self.kwargs['nbits']
        self.increase_factor = self.kwargs['increase_factor']
        assert (0 <= self.increase_factor < 1), 'increase factor ranges in [0, 1)'
        self.expected_bit_sparsity_func = partial(expected_bit_sparsity_func, self.increase_factor)
        self.mask = None
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('init_state', None)
            self.register_buffer('radix_position', None)
            self.register_buffer('weight_int', None)
            self.register_buffer('weight_old', None)
        else:
            self.register_buffer('init_state', torch.zeros(1))
            self.register_buffer('radix_position', torch.zeros(1))
            self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
            self.register_buffer('weight_old', torch.zeros(self.weight.shape))

    def forward(self, input):
        if self.radix_position is None:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:
            il = torch.log2(self.weight.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))

            alpha = 2 ** self.radix_position
            w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
            w_q = w_int / alpha
            self.weight_int.data.copy_(w_int)
            self.weight_old.data.copy_(self.weight)
            if self.training:
                if self.expected_bit_sparsity is None:
                    bit_cnt = count_bit(w_int)
                    original_bit_sparsity = bit_sparse(bit_cnt)
                    self.expected_bit_sparsity = self.expected_bit_sparsity_func(original_bit_sparsity)
                    print('original: {:.3f} expected: {:.3f}'.format(original_bit_sparsity, self.expected_bit_sparsity))
                self.mask = (w_int.abs() > 0).float()
                self.init_state.fill_(1)
        else:
            # quantize weight
            alpha = 2 ** self.radix_position
            w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
            w_q = w_int / alpha
            if self.training:
                bit_cnt_old = count_bit(self.weight_int)
                # bit_sparsity_new = bit_sparse(bit_cnt_new, self.complement)
                bit_sparsity_old = bit_sparse(bit_cnt_old)
                if bit_sparsity_old < self.expected_bit_sparsity:
                    # need bit pruning
                    bit_cnt_new = count_bit(w_int)
                    bit_increase = bit_cnt_new - bit_cnt_old
                    case = (bit_increase > 0)  # todo: bug always False
                    w_q = torch.where(case, self.weight_int.float() * 2 ** (-self.radix_position), w_q)
                    # don't work
                    self.weight.data.copy_(torch.where(case, self.weight_old, self.weight))
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(w_q * 2 ** self.radix_position)
                else:  # don't need bit pruning
                    # print('do not need bit pruning')
                    # use new weights
                    self.weight_old.data.copy_(self.weight)
                    self.weight_int.data.copy_(w_int)
            else:  # inference
                w_q = self.weight_int.data.float() * 2 ** (-self.radix_position)
        # weight has no grad, why? (update optimizer after wrapper.replacement)
        weight_mask = FunctionStopGradient.apply(self.weight, self.mask)
        # weight_mask = self.weight * self.mask
        # STE for quantized weight.
        weight_bp = w_q.detach() + weight_mask - weight_mask.detach()
        out = F.conv2d(input, weight_bp, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        s_prefix = super(Conv2dBPv1, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs)


class ActTruncation(nn.Module):
    def __init__(self, **kwargs_q):
        super(ActTruncation, self).__init__()
        self.kwargs_q = get_default_kwargs_truncation(kwargs_q)
        self.nbits = self.kwargs_q['nbits']
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('radix_position', None)
            self.register_buffer('init_state', None)
        else:
            self.register_buffer('radix_position', torch.zeros(1))
            self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.radix_position is None:
            return x
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0 and x.shape[0] >= 32:  # batch size >=0
            # batch size; remove sqrt[bs] outliers. topk
            batch_size = x.shape[0]
            il = torch.log2(x.abs().view(-1).topk(int(math.sqrt(batch_size)))[0][-1]) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            self.init_state.fill_(1)

        alpha = 2 ** self.radix_position
        x_q = round_pass((x * alpha).clamp(Qn, Qp)) / alpha

        return x_q

    def extra_repr(self):
        s_prefix = super(ActTruncation, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class LinearTruncation(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs_q):
        super(LinearTruncation, self).__init__(in_features, out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_truncation(kwargs_q)
        self.nbits = self.kwargs_q['nbits']
        if 'nbits_a' in self.kwargs_q:
            self.nbits_a = self.kwargs_q['nbits_a']
        else:
            self.nbits_a = 0
        if 0 < self.nbits_a <= 16:
            self.act_truncation = ActTruncation(nbits=self.nbits_a)
        else:
            self.act_truncation = None
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('radix_position', None)
            self.register_buffer('init_state', None)
            self.register_buffer('weight_int', None)
        else:
            self.register_buffer('radix_position', torch.zeros(1))
            self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
            self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.act_truncation is not None:
            x = self.act_truncation(x)
        if self.radix_position is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:
            il = torch.log2(self.weight.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            self.init_state.fill_(1)

        alpha = 2 ** self.radix_position
        w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
        self.weight_int.data.copy_(w_int)
        w_q = w_int / alpha

        return F.linear(x, w_q, self.bias)

    def extra_repr(self):
        s_prefix = super(LinearTruncation, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs_q)


class Conv2dTruncation(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs_q):
        super(Conv2dTruncation, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_truncation(kwargs_q)
        self.nbits = self.kwargs_q['nbits']
        if 'nbits_a' in self.kwargs_q:
            self.nbits_a = self.kwargs_q['nbits_a']
        else:
            self.nbits_a = 0
        if 0 < self.nbits_a <= 16:
            self.act_truncation = ActTruncation(nbits=self.nbits_a)
        else:
            self.act_truncation = None
        if self.nbits > 16 or self.nbits <= 0:
            self.register_buffer('radix_position', None)
            self.register_buffer('init_state', None)
            self.register_buffer('weight_int', None)
        else:
            self.register_buffer('radix_position', torch.zeros(1))
            self.register_buffer('weight_int', torch.zeros(self.weight.shape, dtype=torch.int8))
            self.register_buffer('init_state', torch.zeros(1))

    def forward(self, x):
        if self.act_truncation is not None:
            x = self.act_truncation(x)
        if self.radix_position is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.init_state == 0:
            il = torch.log2(self.weight.abs().max()) + 1
            il = math.ceil(il - 1e-5)
            self.radix_position.data.fill_(self.nbits - il)
            print('Initialize radix position of {} with {}'.format(self._get_name(), int(self.radix_position.item())))
            self.init_state.fill_(1)

        alpha = 2 ** self.radix_position
        w_int = round_pass((self.weight * alpha).clamp(Qn, Qp))
        self.weight_int.data.copy_(w_int)
        w_q = w_int / alpha

        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s_prefix = super(Conv2dTruncation, self).extra_repr()
        return '{}, {}'.format(s_prefix, self.kwargs_q)


bit_code1 = [0, 1, 2, 4, 8, 16, 32, 64]
bit_code1_threshold = [0.0] + [(bit_code1[i] + bit_code1[i + 1]) / 2 for i in range(len(bit_code1) - 1)] + [128]
bit_code2 = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40,
             48, 64, 65, 66, 68, 72, 80, 96]
bit_code2_threshold = [0.0] + [(bit_code2[i] + bit_code2[i + 1]) / 2 for i in range(len(bit_code2) - 1)] + [128]


def bit_pruning_with_truncation(weights, radix_position, log=False, nbits=8):
    scale_factor = 2 ** radix_position
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data_int = linear_quantize_clamp(weights, scale_factor, clamp_min, clamp_max)
    negtive_index = q_data_int < 0
    q_data_int = q_data_int.abs()
    if log:
        bit_code = bit_code1
        bit_code_threshold = bit_code1_threshold
    else:
        bit_code = bit_code2
        bit_code_threshold = bit_code2_threshold

    for i in range(len(bit_code)):
        case = (bit_code_threshold[i] <= q_data_int) * (q_data_int < bit_code_threshold[i + 1])
        q_data_int = torch.where(case, torch.zeros_like(q_data_int).fill_(bit_code[i]), q_data_int)
    q_data_int = torch.where(negtive_index, -q_data_int, q_data_int)
    q_data = linear_dequantize(q_data_int, scale_factor)
    return q_data, q_data_int


class FunctionBitPruningSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, radix_position, log=False):
        weights_bp, w_int = bit_pruning_with_truncation(weights, radix_position, log)
        return weights_bp

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None, None


class FunctionStopGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, stopGradientMask):
        ctx.save_for_backward(stopGradientMask)
        return weight

    @staticmethod
    def backward(ctx, grad_outputs):
        stopGradientMask, = ctx.saved_tensors
        grad_inputs = grad_outputs * stopGradientMask
        return grad_inputs, None


def expected_bit_sparsity_func(factor, input_bit_sparsity):
    return input_bit_sparsity + (1 - input_bit_sparsity) * factor


def count_bit(w_int, complement=False):
    if complement:
        w_int = torch.where(w_int < 0, 256 + w_int, w_int).int().to(w_int.device)
        bit_cnt = torch.zeros(w_int.shape).int().to(w_int.device)
        for i in range(8):
            bit_cnt += w_int % 2
            w_int //= 2
    else:
        w_int = torch.abs(w_int.float()).int()
        bit_cnt = torch.zeros(w_int.shape).int().to(w_int.device)
        for i in range(8):
            bit_cnt += w_int % 2
            w_int //= 2
    return bit_cnt


def bit_sparse(bit_cnt, complement=False):
    if complement:
        return 1 - bit_cnt.sum().float() / (8 * bit_cnt.view(-1).shape[0])
    else:
        return 1 - bit_cnt.sum().float() / (7 * bit_cnt.view(-1).shape[0])
