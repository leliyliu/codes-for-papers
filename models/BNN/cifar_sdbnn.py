
from torch.autograd import Function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math 

__all__ = ['sdresnet20_1w1a', 'sdresnet32_1w1a', 'SDResNet18', 'SDResNet34', 'SD_vgg_small_1w1a']

class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

'''
Reference:
[23] Haotong Qin, Ruihao Gong, Xianglong Liu, Mingzhu Shen,Ziran Wei, Fengwei Yu, and Jingkuan Song. Forward and
backward information retention for accurate binary neural networks. 
In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA,
USA, June 13-19, 2020, pages 2247–2256, 2020.
'''

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.alpha = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        self.dybeta = SELayer(in_channels)

    def forward(self, input):
        w = self.weight
        a = input
        mw = 0.01 * w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1).detach()
        w = w + torch.sigmoid(self.alpha.view(self.alpha.size(0), 1, 1, 1)) * mw
        a = a + self.dybeta(a)
        bw = BinaryQuantize().apply(w, self.k, self.t)
        ba = BinaryQuantize().apply(a, self.k, self.t)

        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        # print(self.beta.grad)
        return output


'''
Reference:
[11] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In 2018 IEEE Conference on Computer Vision and
Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 7132–7141, 2018.
'''
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[o1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear = nn.Linear(512*4*4, num_classes)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        #out = self.softmax(out)
        return out 

def SDResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def SDResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class CBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1w1a, self).__init__()
        self.conv1 = IRConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = IRConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     IRConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        x1 = out
        out = self.bn2(self.conv2(out))
        out += x1
        out = F.hardtanh(out)
        return out

class CResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out


def sdresnet20_1w1a():
    return CResNet(BasicBlock_1w1a, [3, 3, 3])


def resnet20():
    return CResNet(CBasicBlock, [3, 3, 3])

def sdresnet32_1w1a():
    return CResNet(BasicBlock_1w1a, [5, 5, 5])

def resnet32():
    return CResNet(CBasicBlock, [5, 5, 5])


def resnet44():
    return CResNet(CBasicBlock, [7, 7, 7])


def resnet56():
    return CResNet(CBasicBlock, [9, 9, 9])


def resnet110():
    return CResNet(CBasicBlock, [18, 18, 18])


def resnet1202():
    return CResNet(CBasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = IRConv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        # self.nonlinear = nn.ReLU(inplace=True)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = IRConv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = IRConv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = IRConv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = IRConv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

    