"""
    Adaptive Sparsification and Quantization
"""
import math

import scipy.optimize as opt
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from itertools import repeat
from torch._six import container_abcs

from models._modules import _Conv2dQ, Qmodes, _LinearQ, _ActQ

__all__ = ['Conv2dASQ', 'LinearASQ', 'ActASQ']


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_triple = _ntuple(3)


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class SparseSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w):
        w = F.hardshrink(w, 1)
        return w

    @staticmethod
    def backward(ctx, g):
        return g


class Conv2dASQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits=4, nbits_a=4, require_grad=False,
                 mode=Qmodes.layer_wise, standard_threshold=0.5, adaptive=(False, False, False)):
        """
        :param nbits:   the initial quantization bit width of weights.
        :param nbits_a: the initial quantization bit width of activations.
        :param require_grad: whether the weights and bias require to compute gradient.
        :param mode:   Qmodes.layer_wise or Qmodes.kernel_wise
        :param standard_threshold: pruning threshold. pruning ratio = \erf(st/\sqrt(2))
        :param adaptive: whether nbits, standard_threshold; nbits_a are adaptive.
        """
        super(Conv2dASQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits, mode=mode)
        self.add_param('require_grad', require_grad)
        self.weight.requires_grad_(require_grad)
        if bias:
            self.bias.requires_grad_(require_grad)
        self.add_param('nbits_a', nbits_a)
        adaptive = _triple(adaptive)
        self.add_param('adaptive', adaptive[:2])
        if self.kwargs_q['nbits_a'] > 0:
            self.act_sq = ActASQ(nbits=self.kwargs_q['nbits_a'], adaptive=adaptive[2])
        else:
            self.act_sq = None
        if standard_threshold > 0:
            self.standard_threshold = Parameter(torch.Tensor([standard_threshold]), requires_grad=adaptive[1])
        else:
            self.register_parameter('standard_threshold', None)
        self.nbits_param = Parameter(torch.Tensor([nbits]),
                                     requires_grad=adaptive[0])
        # todo: backward??? override self.nbits

    def forward(self, x):
        # Sparsification
        # https://www.shuxuele.com/data/standard-normal-distribution-table.html
        # Reference: https://github.com/rhhc/SparsityLoss/blob/master/sparse_utils.py
        weight_sparse = self.weight
        if self.standard_threshold:
            standard_threshold = torch.clamp(self.standard_threshold, 0, 5)
            with torch.no_grad():
                self.standard_threshold.data.copy_(standard_threshold)
            l = standard_threshold * self.weight.std().detach()
            nw = self.weight / (l + 1e-6)
            sw = SparseSTE.apply(nw)  # todo: is there any gradient ????
            weight_sparse = (l + 1e-6) * sw

        if self.act_sq is not None:
            x = self.act_sq(x)

        if self.alpha is None:
            return F.conv2d(x, weight_sparse, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        if self.q_mode == Qmodes.layer_wise:
            w_reshape = weight_sparse
        else:
            w_reshape = weight_sparse.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        self.nbits = round_pass(self.nbits_param).clamp(2, 8)
        Qn = (-2 ** (self.nbits - 1)).detach()
        Qp = (2 ** (self.nbits - 1) - 1).detach()
        if self.init_state == 0:
            if self.q_mode == Qmodes.layer_wise:
                if self.nbits >= 7:
                    self.alpha.data.copy_(weight_sparse.detach().abs().max() / Qp)
                else:
                    self.alpha.data.copy_(2 * weight_sparse.detach().abs().mean() / math.sqrt(Qp))
                    # self.alpha.data.copy_(get_alpha(self.weight, Qn, Qp))
                print('initialize alpha {:.4f} of {}'.format(self.alpha.item(), self._get_name()))
            else:
                self.alpha.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / Qp)
                # if self.nbits >= 7:
                #     self.alpha.data.copy_(w_reshape.detach().abs().max(dim=0)[0] / Qp)
                # else:
                #     self.alpha.data.copy_(w_reshape.abs().mean(dim=0) / math.sqrt(Qp))
                print('initialize alpha mean {:.4f} of {}'.format(self.alpha.mean().item(), self._get_name()))
            self.init_state.fill_(1)
            return F.conv2d(x, weight_sparse, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        """  
        Implementation according to paper. 
        """
        g = 1.0 / math.sqrt(weight_sparse.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        clip_max = 1 - 2 ** (1 - self.nbits.detach()).item()
        w_q = round_pass(
            (w_reshape / alpha * 2 ** (1 - self.nbits)).clamp(-1, clip_max) * 2 ** (
                    self.nbits - 1)) * alpha
        # w = w.clamp(Qn, Qp) #todo: kernel-wise quantization
        # q_w = round_pass(w)
        # w_q = q_w * alpha
        if not self.q_mode == Qmodes.layer_wise:
            w_q = w_q.transpose(0, 1).reshape(weight_sparse.shape)
        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LinearASQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits=4):
        super(LinearASQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias, nbits=nbits)

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


class ActASQ(_ActQ):
    def __init__(self, nbits=4, adaptive=False):
        """
        :param nbits: the initial quantization bit width of activation
        :param adaptive: whether the nbits of activations is adaptive.
        """
        super(ActASQ, self).__init__(nbits=nbits, signed=False)
        self.add_param('adaptive', adaptive)
        self.nbits_param = Parameter(torch.Tensor([nbits]),
                                     requires_grad=adaptive)

    def forward(self, x):
        if self.alpha is None:
            return x
        # if self.signed:
        #     Qn = -2 ** (self.nbits - 1)
        #     Qp = 2 ** (self.nbits - 1) - 1
        # else:
        #     Qn = 0
        #     Qp = 2 ** self.nbits - 1
        self.nbits = round_pass(self.nbits_param).clamp(2, 8)
        if x.min() > -1e-5:
            Qn = 0
            Qp = (2 ** self.nbits - 1).detach().item()
            scale_b = 2 ** self.nbits
            self.signed = False
        else:
            Qn = (-2 ** (self.nbits - 1)).detach().item()
            Qp = (2 ** (self.nbits - 1) - 1).detach().item()
            scale_b = 2 ** (self.nbits - 1)
            self.signed = True
        if self.init_state == 0 and x.shape[0] >= 32:
            # The init alpha for activation is very very important as the experimental results shows.
            # Please select a init_rate for activation.
            # if self.nbits >= 7:
            # topk (10)
            self.alpha.data.copy_(get_alpha(x, Qn, Qp))
            print('initialize alpha {:.4f} of ActASQ'.format(self.alpha.item()))
            # else:
            #     self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            self.init_state.fill_(1)
            # if we return x directly, alpha will get no grad when backward.
            return x

        g = 1.0 / math.sqrt(x.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha / scale_b).clamp(Qn / scale_b.detach().item(),
                                                   Qp / scale_b.detach().item()) * scale_b) * alpha
        # x = x / alpha
        # x = x.clamp(Qn, Qp)
        # q_x = round_pass(x)
        # x_q = q_x * alpha

        # Method2:
        # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        return x


def mse(x, alpha, Qn, Qp):  # todo: speedup
    alpha = torch.from_numpy(alpha).to(x.device)
    x_clip = (x / alpha).clamp(Qn, Qp)
    x_q = x_clip.round()
    x_q = x_q * alpha
    return (((x_q - x) ** 2).sum() / x.numel()).cpu().item()


def get_alpha(x, Qn, Qp):
    # method1
    alpha = x.view(x.shape[0], -1).max(axis=1)[0].topk(10)[0][-1] / Qp
    # mse0 = mse(x, alpha.cpu().numpy(), Qn, Qp)
    mmse = lambda param: mse(x, param, Qn=Qn, Qp=Qp)
    res = opt.minimize(mmse, (alpha.detach().cpu().numpy()), method='powell',
                       options={'disp': False, 'ftol': 0.05, 'maxiter': 100, 'maxfev': 100})
    return torch.from_numpy(res.x).abs()
