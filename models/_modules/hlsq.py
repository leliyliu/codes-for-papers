"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
import math
from models._modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ

__all__ = ['Conv2dHSQ', 'LinearHSQ', 'ActHSQ']


class FunLSQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp
        q_w = (weight / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = weight / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that alpha is always greater than zero in any case and can also
        # suppress the update speed of alpha. (Personal understanding)
        # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
        return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dHSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4,
                 mode=Qmodes.layer_wise):
        super(Conv2dHSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        self.act = ActHSQ(nbits=nbits, signed=False)

    def get_quan_filters(self, filters):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * filters.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(filters.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        w = filters / alpha.abs()
        w = F.hardtanh(w, Qn, Qp)
        w_q = round_pass(w) * alpha.abs()
        return w_q

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        w_q = self.get_quan_filters(self.weight)
        x = self.act(x)
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class LinearHSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearHSQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        return F.linear(x, w_q, self.bias)

class ActHSQ(_ActQ):
    def __init__(self, nbits=4, signed=False):
        super(ActHSQ, self).__init__(nbits=nbits, signed=signed)

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
            self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = x / alpha.abs()
        x = F.hardtanh(x, Qn, Qp)
        x_q = round_pass(x) * alpha.abs()

        return x_q
