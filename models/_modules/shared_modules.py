import torch.nn as nn
import torch.nn.functional as F
import torch
from models._modules import k_means_cpu, reconstruct_weight_from_k_means_result, FuncKmeansSTE
import ipdb

__all__ = ['Conv2dShare', 'BatchNorm2dShare', 'ReLUshare', 'MaxPool2dShare', 'ActShare', 'Conv2dShareW']


class Conv2dShare(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, share_num=2):
        super(Conv2dShare, self).__init__()
        self.share_num = share_num
        self.convs = nn.ModuleList()
        for i in range(share_num):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups, bias=bias))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.convs[i](input[i]))
        return ret


class Conv2dShareW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, share_num=2, nbits=-1, share_ratio=0.0):
        super(Conv2dShareW, self).__init__()
        self.share_num = share_num
        self.nbits = nbits
        assert 0.0 <= share_ratio <= 1.0
        self.share_ratio = share_ratio

        self.convs = nn.ModuleList()
        for i in range(share_num):
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups, bias=bias))
        print(self.convs[0].weight.shape)
        self.share_split = int(out_channels * share_ratio)
        print('share part: {}/{}'.format(self.share_split, out_channels))
        if nbits < 0:
            self.nbits = nbits
            self.register_buffer('centroids', None)
            self.register_buffer('labels', None)
            return
        self.nbits = nbits
        self.mode = 'cpu'
        self.centroids = nn.Parameter(torch.zeros(2 ** self.nbits))

        self.register_buffer('labels',
                             torch.zeros(self.share_split + self.share_num * (out_channels - self.share_split),
                                         self.convs[0].weight.size(1), self.convs[0].weight.size(2),
                                         self.convs[0].weight.size(3)))
        self.register_buffer('labels_split',
                             torch.zeros(
                                 self.convs[0].weight.size(0) * 3,
                                 self.convs[0].weight.size(1), self.convs[0].weight.size(2),
                                 self.convs[0].weight.size(3)
                             ))
        self.register_buffer('init_state', torch.zeros(1))

    def forward(self, input):
        share_weight = self.convs[0].weight[:self.share_split, :, :, :]
        if self.nbits < 0:
            ret = []
            for i in range(self.share_num):
                ret.append(
                    F.conv2d(input[i], torch.cat((share_weight, self.convs[i].weight[self.share_split:, :, :, :])),
                             self.convs[i].bias, self.convs[i].stride,
                             self.convs[i].padding, self.convs[i].dilation, self.convs[i].groups))
            return ret
        split = int(self.convs[0].weight.size(0) - self.share_split)
        if self.init_state == 0 and self.training:
            if self.mode == 'cpu':
                with torch.no_grad():
                    weight = [share_weight]
                    for i in range(self.share_num):
                        weight.append(self.convs[i].weight[self.share_split:, :, :, :].data)
                    weight = torch.cat(weight)
                    centroids, labels = k_means_cpu(weight.cpu().numpy(), 2 ** self.nbits)
                    self.centroids.copy_(centroids)
                    self.labels.copy_(labels)
                    wq = reconstruct_weight_from_k_means_result(centroids, labels)
                    share_weight_q = wq[: self.share_split, :, :, :]
                    self.convs[0].weight[:self.share_split, :, :, :].data.copy_(share_weight_q)
                    for i in range(self.share_num):
                        wqi = wq[self.share_split + i * split: self.share_split + (i + 1) * split, :, :, :]
                        self.convs[i].weight[self.share_split:, :, :, :].data.copy_(wqi)
                print('conv kmeans processing.')
            else:
                raise NotImplementedError
            self.init_state.fill_(1)
        weight = [share_weight]
        for i in range(self.share_num):
            weight.append(self.convs[i].weight[self.share_split:, :, :, :].data)
            if True:  # todo: generate data for simulator (to be del)
                out_channels = self.convs[0].out_channels
                private_split = out_channels - self.share_split
                self.convs[i].weight[:self.share_split, :, :, :].data.copy_(share_weight)
                self.labels_split[i * out_channels: (i + 1) * out_channels, :, :, :].copy_(
                    torch.cat([self.labels[:self.share_split, :, :, :],
                               self.labels[
                               self.share_split + private_split * i: self.share_split + private_split * (i + 1),
                               :, :, :]])
                )
        weight = torch.cat(weight)
        weight_q = FuncKmeansSTE.apply(weight, self.centroids, self.labels)
        ret = []
        share_weight_q = weight_q[: self.share_split, :, :, :]
        for i in range(self.share_num):
            wqi = weight_q[self.share_split + i * split: self.share_split + (i + 1) * split, :, :, ]
            self.convs[i].weight[self.share_split:, :, :, :].data.copy_(wqi)
            ret.append(
                F.conv2d(input[i], torch.cat((share_weight_q, wqi)),
                         self.convs[i].bias, self.convs[i].stride,
                         self.convs[i].padding, self.convs[i].dilation, self.convs[i].groups))

        return ret

    def extra_repr(self):
        s_prefix = super(Conv2dShareW, self).extra_repr()
        return '{}, nbits={}, share_num={}, share_ratio {}'.format(
            s_prefix, self.nbits, self.share_num, self.share_ratio)


class ActShare(nn.Module):
    def __init__(self):
        super(ActShare, self).__init__()

    def forward(self, input):
        return input


class BatchNorm2dShare(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, share_num=2):
        super(BatchNorm2dShare, self).__init__()
        self.share_num = share_num
        self.bns = nn.ModuleList()
        for i in range(share_num):
            self.bns.append(nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.bns[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(BatchNorm2dShare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)


class ReLUshare(nn.Module):
    def __init__(self, inplace=False, share_num=2):
        super(ReLUshare, self).__init__()
        self.share_num = share_num
        self.relu_list = nn.ModuleList()
        for i in range(share_num):
            self.relu_list.append(nn.ReLU(inplace))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.relu_list[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(ReLUshare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)


class MaxPool2dShare(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, share_num=2):
        super(MaxPool2dShare, self).__init__()
        self.share_num = share_num
        self.pools = nn.ModuleList()
        for i in range(share_num):
            self.pools.append(nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode))

    def forward(self, input):
        ret = []
        for i in range(self.share_num):
            ret.append(self.pools[i](input[i]))
        return ret

    def extra_repr(self):
        s_prefix = super(MaxPool2dShare, self).extra_repr()
        return '{}, share_num={}'.format(
            s_prefix, self.share_num)
