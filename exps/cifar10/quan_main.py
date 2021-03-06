'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time 
import logging
import sys
import argparse
import numpy as np
import random

import models.cifar10.fp as models
from utils.ptflops import get_model_complexity_info
from wrapper.qcode_wrapper import replace_conv_recursively

q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Quantization')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--q-mode', choices=q_modes_choice, default='layer_wise',
                    help='Quantization modes: ' +
                            ' | '.join(q_modes_choice) +
                            ' (default: kernel-wise)')
parser.add_argument('--quan-mode', type=str, default='Conv2dLSQ', 
                    help='corresponding for the quantize conv')
parser.add_argument('--evaluate', default=False, action='store_true', 
                    help='evaluate for model')
parser.add_argument('--save', type=str, default='EXP', 
                    help='path for saving trained models')
parser.add_argument('--manual-seed', default=2, type=int, help='random seed is settled')
args = parser.parse_args()

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)  # 设置随机种子

args.save = 'cifar10-{}-{}-{}-{}'.format(args.quan_mode[-3:], args.arch, args.save, time.strftime("%Y%m%d-%H%M%S"))

from tensorboardX import SummaryWriter
writer_comment = args.save 
log_dir = '{}/{}'.format('logger', args.save)
writer = SummaryWriter(log_dir = log_dir, comment=writer_comment)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join('{}/log.txt'.format(log_dir)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
net = models.__dict__[args.arch]()

# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/{}-ckpt.pth'.format(args.arch))
net.load_state_dict(checkpoint['net'])

net = replace_conv_recursively(net, args.quan_mode, args)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from Quantized checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-{}-ckpt.pth'.format(args.arch, args.quan_mode[-3:]))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.quan_mode != 'Conv2dDPQ':
    flops, params = get_model_complexity_info(net, (3,32,32))
    print('the total flops of mobilenetv2 is : {} and whole params is : {}'.format(flops, params)) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    logging.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    logging.info('the test acc is : {}'.format(acc))
    if acc > best_acc:
        logging.info('Saving..')
        if args.quan_mode == 'Conv2dDPQ':
            flops, params = get_model_complexity_info(net, (3,32,32))
            logging.info('the total flops of {} is : {} and whole params is : {}'.format(args.arch, flops, params)) 
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}-{}-ckpt.pth'.format(args.arch, args.quan_mode[-3:]))
        best_acc = acc


if args.evaluate:
    test(start_epoch)
else:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()

    logging.info('after {} train for quantization, the best acc1 is {} '.format(args.quan_mode[-3:], best_acc))