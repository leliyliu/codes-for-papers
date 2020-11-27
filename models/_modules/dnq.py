import math

import torch
import torch.nn.functional as F

from models._modules import _ActQ, log_shift, _Conv2dQ, Qmodes, _LinearQ, ln_error, update_running_scale

__all__ = ['ActDNQ', 'Conv2dDNQ', 'LinearDNQ',
           'ActDNQv2', 'Conv2dDNQv2', 'LinearDNQv2']


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


def grad_shift_scale(x, scale):
    y = x
    y_grad = x * scale
    y = log_shift(y)
    return y.detach() - y_grad.detach() + y_grad


class ActDNQ(_ActQ):
    def __init__(self, nbits=4, signed=False):
        super(ActDNQ, self).__init__(nbits=nbits, signed=signed)
        self.register_buffer('alpha_s', torch.zeros(1))
        self.fixed = False

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            # self.alpha /= 2 # 1 no
            self.alpha /= (2 ** (self.nbits - 1))
            self.fixed = True
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None or x.max() < 1e-6:
            assert ValueError
            return x
        if self.signed:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        if self.training and self.init_state == 0:
            # Please select a init_rate for activation.
            alpha_fp = 2 * x.abs().mean() / math.sqrt(Qp)
            self.alpha.data.copy_(alpha_fp)
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_shift_scale(self.alpha, g)
        if self.fixed:
            alpha = alpha.detach()
        self.alpha_s.data.copy_(alpha)
        x_q = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        return x_q


class LinearDNQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearDNQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)
        self.register_buffer('alpha_s', torch.zeros(1))
        self.fixed = False

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            # self.alpha /= 2 # 1 no
            self.alpha /= (2 ** (self.nbits - 1))
            self.fixed = True
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            # alpha_fp = self.weight.detach().abs().max() / (Qp + 1)
            alpha_fp = 2 * self.weight.abs().mean() / (Qp ** 0.5)
            # print('{}==>{}'.format(alpha_fp.item(), alpha_s.item()))
            self.alpha.data.copy_(alpha_fp)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_shift_scale(self.alpha, g)
        if self.fixed:
            alpha = alpha.detach()
        self.alpha_s.data.copy_(alpha)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return F.linear(x, w_q, self.bias)


class Conv2dDNQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise):
        super(Conv2dDNQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        self.register_buffer('alpha_s', torch.zeros(1))
        self.fixed = False

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            # self.alpha /= 2 # 1 no
            self.alpha /= (2 ** (self.nbits - 1))
            self.fixed = True
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            if self.q_mode == Qmodes.layer_wise:
                # alpha_fp = w_reshape.detach().abs().max() / (Qp + 1)
                alpha_fp = 2 * self.weight.abs().mean() / (Qp ** 0.5)
            else:
                assert NotImplementedError
                # alpha_fp = w_reshape.detach().abs().max(dim=0)[0] / Qp
                # alpha_s = log_shift(alpha_fp)
                print('-----')
            self.alpha.data.copy_(alpha_fp)
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)
        alpha = grad_shift_scale(self.alpha, g)
        if self.fixed:
            alpha = alpha.detach()
        self.alpha_s.data.copy_(alpha)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w_reshape_q = (w_reshape / alpha).round().clamp(Qn, Qp) * alpha
        # w_q = w_reshape_q.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class FunLLSQS(torch.autograd.Function):
    # TODO:
    @staticmethod
    def forward(ctx, x, alpha, Qn, Qp, Qmode, is_l2, is_act=True):
        ctx.other = Qn, Qp, Qmode, is_l2, is_act
        alpha = log_shift(alpha)
        q_x = (x / alpha).round().clamp(Qn, Qp)
        x_q = q_x * alpha
        ctx.save_for_backward(x, alpha)
        return x_q

    @staticmethod
    def backward(ctx, grad_x):
        x, alpha = ctx.saved_tensors
        if alpha.shape[0] == 1:
            assert alpha > 0, 'alpha = {}'.format(alpha)
        else:
            assert (alpha >= 0).sum() == alpha.shape[0]
        Qn, Qp, Qmode, is_l2, is_act = ctx.other
        if is_act:
            zeros_x = torch.zeros_like(grad_x).to(alpha.device)
            grad_x = torch.where(((Qn * alpha < x) + (x < Qp * alpha)) == 2, grad_x, zeros_x)
        error = ln_error(x, alpha, Qn, Qp, Qmode, is_l2)
        b, s = update_running_scale(x, alpha, error, Qn, Qp, Qmode, is_l2)
        grad_alpha = torch.zeros_like(alpha).to(alpha.device)
        grad_alpha = torch.where(b, -(alpha ** 2), grad_alpha) + \
                     torch.where(s, alpha ** 2, grad_alpha)
        return grad_x, grad_alpha, None, None, None, None, None


class ActDNQv2(_ActQ):
    def __init__(self, nbits=4, signed=False, is_l2=True):
        super(ActDNQv2, self).__init__(nbits=nbits, signed=signed)
        self.add_param('is_l2', is_l2)

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None:
            return x
        if self.signed:
            Qn = -2 ** (self.nbits - 1)
            Qp = 2 ** (self.nbits - 1) - 1
        else:
            Qn = 0
            Qp = 2 ** self.nbits - 1
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            # empirical value
            if self.nbits >= 4:
                init_value = (Qp + 1)
            elif self.nbits == 3:
                init_value = 2 * Qp
            else:
                # TODO
                init_value = 2 * Qp
            self.alpha.data.fill_(x.detach().abs().max() / init_value)

        y = FunLLSQS.apply(x, self.alpha, Qn, Qp, Qmodes.layer_wise, self.kwargs_q['is_l2'], True)
        return y


class LinearDNQv2(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4, is_l2=True):
        super(LinearDNQv2, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)
        self.add_param('is_l2', is_l2)

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.transpose(0, 1)
        if self.training and self.init_state == 0:
            self.init_state.fill_(1)
            self.alpha.data.fill_(w_reshape.detach().abs().max() / (Qp + 1))
        w_reshape_q = FunLLSQS.apply(w_reshape, self.alpha, Qn, Qp, Qmodes.layer_wise, self.kwargs_q['is_l2'], False)
        w_q = w_reshape_q.transpose(0, 1)
        return F.linear(x, w_q, self.bias)


class Conv2dDNQv2(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise, is_l2=True):
        super(Conv2dDNQv2, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        self.add_param('is_l2', is_l2)

    def set_dnq(self, nbits=None):
        if nbits is not None:
            self.nbits = nbits
        else:
            self.nbits = 2 * self.nbits

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        if self.training and self.init_state == 0:
            if self.q_mode == Qmodes.layer_wise:
                self.alpha.data.copy_(w_reshape.detach().abs().max() / (Qp + 1))
            else:
                self.alpha.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / Qp)
            self.init_state.fill_(1)
        w_reshape_q = FunLLSQS.apply(w_reshape, self.alpha, Qn, Qp, self.q_mode, self.kwargs_q['is_l2'], False)
        w_q = w_reshape_q.transpose(0, 1).reshape(self.weight.shape)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
