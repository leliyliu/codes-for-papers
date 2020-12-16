"""
    Adaptive Sparsification and Quantization
"""
import math

import scipy.optimize as opt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from itertools import repeat
from torch._six import container_abcs

from models._modules import Qmodes

__all__ = ['Conv2dDPQ', 'linearDPQ', 'ActDPQ']


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def quantize_pow2(v):
    return 2 ** round_pass((torch.log(v) / math.log(2.)))

class FunDPQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, qmax, g):
        assert alpha > 0, 'alpha = {}'.format(alpha)
        ctx.save_for_backward(weight, alpha, qmax)
        ctx.other = g
        xmin, xmax = -qmax.item(), qmax.item()
        q_w = (weight / alpha).round().clamp(xmin, xmax)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, qmax = ctx.saved_tensors
        g = ctx.other
        xmin, xmax = -qmax.item(), qmax.item()
        q_w = weight / alpha
        indicate_small = (q_w < -xmin).float()
        indicate_big = (q_w > xmax).float()
        indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
        grad_alpha = ((indicate_middle * (-q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = indicate_middle * grad_weight
        grad_qmax = indicate_big - indicate_small
        return grad_weight, grad_alpha, grad_qmax, None

class Conv2dDPQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
                 qmin=1e-3, qmax=100, dmin=1e-5, dmax=10, bias=True, sign=True, wbits=4, abits=4, mode=Qmodes.layer_wise):
    
        """
        :param d_init: the inital quantization stepsize (alpha)
        :param mode: Qmodes.layer_wise or Qmodes.kernel_wise
        :param xmax_init: the quantization range for whole weights 
        """

        super(Conv2dDPQ, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.qmin = qmin
        self.qmax = qmax
        self.dmin = dmin 
        self.dmax = dmax
        self.q_mode = mode
        self.sign = sign
        self.nbits = wbits 
        self.act_dpq = ActDPQ(signed=False, nbits=abits)
        self.alpha = Parameter(torch.Tensor(1))
        self.xmax = Parameter(torch.Tensor(1))
        self.weight.requires_grad_(True)
        if bias:
            self.bias.requires_grad_(True)
        self.register_buffer('init_state', torch.zeros(1))

    def get_nbits(self):
        abits = self.act_dpq.get_nbits()
        xmax = self.xmax.abs().item()
        alpha = self.alpha.abs().item()
        if self.sign:
            nbits = math.ceil(math.log(xmax/alpha + 1) / math.log(2) + 1)
        else:
            nbits = math.cell(math.log(xmax/alpha + 1) / math.log(2))

        self.nbits = nbits
        return abits, nbits

    def get_quan_filters(self, filters):
        if self.training and self.init_state == 0:
            Qp = 2 ** (self.nbits - 1) - 1
            self.xmax.data.copy_(filters.abs().max())
            self.alpha.data.copy_(self.xmax / Qp)
            # self.alpha[self.index].data.copy_(2 * filters.abs().mean() / math.sqrt(Qp))
            # self.xmax[self.index].data.copy_(self.alpha[self.index] * Qp)
            self.init_state.fill_(1)

        Qp = (self.xmax.detach()/self.alpha.detach()).abs().item()
        g = 1.0 / math.sqrt(filters.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        xmax = grad_scale(self.xmax, g)

        w = F.hardtanh(filters/xmax.abs(), -1, 1) * xmax.abs()
        w = w/alpha.abs()
        wq = round_pass(w)*alpha.abs()

        return wq

    def forward(self,x):
        if self.act_dpq is not None:
            x = self.act_dpq(x)
        
        wq = self.get_quan_filters(self.weight)
        return F.conv2d(x, wq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class linearDPQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, d_min=2**-8, d_max=2**8, xmax_min=0.001, xmax_max=10):
        super(linearDPQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.d_min = d_min
        self.d_max = d_max 
        self.xmax_min = xmax_min
        self.xmax_max = xmax_max
        self.alpha = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

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

class ActDPQ(nn.Module):
    def __init__(self, signed=False, nbits=4, qmin=1e-3, qmax=100, dmin=1e-5, dmax=10):
        """
        :param nbits: the initial quantization bit width of activation
        :param signed: whether the activation data is signed
        """
        super(ActDPQ, self).__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.dmin = dmin 
        self.dmax = dmax
        self.signed = signed
        self.nbits = nbits
        self.index = 0
        self.alpha = Parameter(torch.Tensor(1))
        self.xmax = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def get_nbits(self):
        self.xmax.data.copy_(self.xmax.clamp(self.qmin, self.qmax))
        self.alpha.data.copy_(self.alpha.clamp(self.dmin, self.dmax))
        if self.signed:
            nbits = (torch.log(self.xmax/self.alpha + 1) / math.log(2) + 1).ceil()
        else:
            nbits = (torch.log(self.xmax/self.alpha + 1) / math.log(2)).ceil()
        self.nbits = int(nbits.item())
        return nbits

    def forward(self, x):
        if self.alpha is None:
            return x
        
        if self.training and self.init_state == 0:
            Qp = 2 ** (self.nbits - 1) - 1
            self.xmax.data.copy_(x.abs().max())
            self.alpha.data.copy_(self.xmax / Qp)
            # self.alpha[self.index].data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
            # self.xmax[self.index].data.copy_(self.alpha[self.index] * Qp)
            self.init_state.fill_(1)

        self.xmax.data.copy_(self.xmax.clamp(self.qmin, self.qmax))
        self.alpha.data.copy_(self.alpha.clamp(self.dmin, self.dmax))
        Qp = (self.xmax/self.alpha).item()
        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        xmax = grad_scale(self.xmax, g)

        if self.signed: 
            x = F.hardtanh(x/xmax.abs(), -1, 1) * xmax.abs()
            # x = round_pass((torch.clamp(x/xmax, -1, 1)*xmax)/alpha) * alpha
        else:
            x = F.hardtanh(x/xmax.abs(), 0, 1) * xmax.abs()
            # x = round_pass((torch.clamp(x/xmax, 0, 1)*xmax)/alpha) * alpha
        x = x / alpha.abs()
        x = round_pass(x) * alpha.abs()
        
        return x

def mse(x, alpha, sign, xmax):  
    alpha = torch.from_numpy(alpha).to(x.device)
    if sign:
        x_clip = (x / alpha).clamp(0, xmax)
    else:
        x_clip = (x / alpha).clamp(-xmax, xmax)
    x_q = x_clip.round()
    x_q = x_q * alpha
    return (((x_q - x) ** 2).sum() / x.numel()).cpu().item()

def get_alpha(x, sign, xmax):
    # method1
    # print('the x shape is : ' , x.shape)
    alpha = x.view(x.shape[0], -1).max(axis=1)[0].topk(10)[0][-1] / xmax

    mmse = lambda param: mse(x, param, sign=sign, xmax=xmax)
    res = opt.minimize(mmse, (alpha.detach().cpu().numpy()), method='powell',
                       options={'disp': False, 'ftol': 0.05, 'maxiter': 100, 'maxfev': 100})
    return torch.from_numpy(res.x).abs()
