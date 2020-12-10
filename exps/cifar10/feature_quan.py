'''Train CIFAR10 with PyTorch.'''
from math import log
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
from exps.cifar10.run_tool import add_weight_decay, initial_blocks, finetune, validate
from utils.network import save_checkpoint


q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
lr_scheduler_choice = ['StepLR', 'MultiStepLR', 'CosineAnnealingLR']
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Quantization')
parser.add_argument('-b', '--batch-size', help='The batch on every device for validation',
    type=int, default=128)
parser.add_argument('-j', '--workers', help='Number of workers', type=int, default=20)
parser.add_argument('-a', '--arch', metavar='ARCH', default='MobileNetV2',
                    help='model architecture (default: MobileNetV2)')
parser.add_argument('--teacher-arch', metavar='TEACHER ARCH', default='ResNet18',
                    help='teacher model architecture (default: ResNet18)')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--q-mode', choices=q_modes_choice, default='layer_wise',
                    help='Quantization modes: ' +
                            ' | '.join(q_modes_choice) +
                            ' (default: kernel-wise)')
parser.add_argument('--quan-mode', type=str, default='Conv2dLSQ', 
                    help='corresponding for the quantize conv')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--evaluate', default=False, action='store_true', 
                    help='evaluate for model')
parser.add_argument('--save', type=str, default='EXP', 
                    help='path for saving trained models')
parser.add_argument('--manual-seed', default=2, type=int, help='random seed is settled')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument("--kd-ratio", type=float, default=0.5, help="learning from soft label distribution")
parser.add_argument('--feature-ratio', type=float, default=0.4, help="learning from teacher model feature map")
parser.add_argument('--epochs', type=int, default=90, help='the whole train epoch')
parser.add_argument('--initial-epochs', type=int, default=10, help='initial epochs for specific blocks')
parser.add_argument('--initial-lr', default=2e-3, type=float, help='learning rate for initial blocks')
parser.add_argument('--lr-scheduler', default='CosineAnnealingLR', choices=lr_scheduler_choice)
parser.add_argument('--initial-scheduler', default='CosineAnnealingLR', choices=lr_scheduler_choice)
parser.add_argument('--save_model', type=str, default='cifar-MobileNetV2-HSQ-models/model_best.pth.tar')
args = parser.parse_args()

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
np.random.seed(args.manual_seed)
random.seed(args.manual_seed)  # 设置随机种子

args.save = 'cifar10-feature-{}-{}-{}-{}'.format(args.quan_mode[-3:], args.arch, args.save, time.strftime("%Y%m%d-%H%M%S"))

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
logging.info('==> Preparing data..')
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
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def main():
# Model
    logging.info('==> Building model {} ..'.format(args.arch))

    net = models.__dict__[args.arch]()

    stage_blocks = []
    block = ['first_']
    for i in range(len(net.layers)):
        if net.layers[i].conv2.stride == 2:
            stage_blocks.append(block)
            block = []
        block.append('layers.{}'.format(i))   

    stage_blocks.append(block)
    last_block = ['last_', 'linear']

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Load checkpoint.
    if args.pretrained:
        logging.info('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.pth'.format(args.arch))
        net.load_state_dict(checkpoint['net'])

    net = replace_conv_recursively(net, args.quan_mode, args)

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        validate_loss, validate_top1 = validate(net, criterion, data_loader=testloader, epoch=0, valid=True)
        return 

    if args.kd_ratio > 0: # 设置teacher model (supernet)，实际上是最大的网络
        args.teacher_model = models.__dict__[args.teacher_arch]()
        args.teacher_model = torch.nn.DataParallel(args.teacher_model)
        checkpoint = torch.load('./checkpoint/{}-ckpt.pth'.format(args.teacher_arch))
        args.teacher_model.load_state_dict(checkpoint['net'])

    best_acc1 = 0
    args.initial_start_epoch, args.start_epoch = 0, 0
    if args.resume:
        # Load checkpoint.
        logging.info('==> Resuming from Quantized checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('logger/{}'.format(args.save_model))
        net.load_state_dict(checkpoint['net'])
        best_acc1 = checkpoint['acc']
        if 'initial' in args.save_model:
            args.initial_start_epoch = checkpoint['initial_epoch'] + 1
        else:
            args.initial_start_epoch = args.initial_epochs 
            args.start_epoch = checkpoint['epoch'] + 1

    if args.quan_mode != 'Conv2dDPQ':
        flops, params = get_model_complexity_info(net, (3,32,32))
        logging.info('the total flops of {} is : {} and whole params is : {}'.format(args.arch, flops, params)) 

    optimizers, schedulers = [], []
    for block in stage_blocks:
        params = add_weight_decay(net, weight_decay=args.weight_decay, skip_keys=['expand_','running_scale'], grads=block)
        optimizer = torch.optim.SGD(params, args.initial_lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.initial_epochs)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    params = add_weight_decay(net, weight_decay=args.weight_decay, skip_keys=['expand_', 'running_scale'], grads=last_block)
    final_optimizer = optim.SGD(params, args.initial_lr, momentum=args.momentum)
    final_scheduler = optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=args.initial_epochs)

    fin_optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    fin_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fin_optimizer, T_max=200)

    with open('cifar-feature-quan.txt', 'w+') as f:
        for epoch in range(args.initial_start_epoch, args.initial_epochs):
            initial_blocks(net, args.teacher_model, trainloader, optimizers, epoch)
            train_loss, train_top1 = finetune(net, args.teacher_model, args, trainloader, final_optimizer, epoch, criterion)
            validate_loss, validate_top1 = validate(net, criterion, data_loader=testloader, epoch=epoch, valid=True)
            logging.info('the soft train loss is : {} ; For Train !  the top1 accuracy is : {} ;'.format(train_loss, train_top1))
            logging.info('the validate loss is : {} ; For Validate !  the top1 accuracy is : {} ;'.format(validate_loss, validate_top1))
            writer.add_scalars('Inital-block-Loss/Training-Validate',{
                'train_soft_loss': train_loss,
                'validate_loss': validate_loss
            }, epoch + 1)
            writer.add_scalars('Inital-block-Top1/Training-Validate',{
                'train_acc1': train_top1,
                'validate_acc1': validate_top1
            }, epoch + 1)
            writer.add_scalars('Learning-Rate-For-Initial', {
                'basic optimizer': final_optimizer.state_dict()['param_groups'][0]['lr'],
            }, epoch + 1)
            is_best = False
            if validate_top1 > best_acc1:
                best_acc1 = validate_top1
                is_best = True
                logging.info('the best model top1 is : {} and its epoch is {} !'.format(best_acc1, epoch))
            save_checkpoint({
                'initial_epoch': epoch,
                'state_dict': net.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': final_optimizer.state_dict(),
                'scheduler': final_scheduler.state_dict(),
            }, is_best, save='logger/{}-{}-{}-initial-models'.format('cifar', args.arch, args.quan_mode[-3:]))
            for scheduler in schedulers:
                scheduler.step()
            final_scheduler.step()
        logging.info('After initial from feature, we can get the final soft train loss and the top1 accuracy')
        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_top1 = finetune(net, args.teacher_model, args, trainloader, fin_optimizer, epoch, criterion)
            validate_loss, validate_top1 = validate(net, criterion, data_loader=testloader, epoch=epoch, valid=True) 
            logging.info('the train loss is : {} ; For Train !  the top1 accuracy is : {} ;'.format(train_loss, train_top1))
            logging.info('the validate loss is : {} ; For Validate !  the top1 accuracy is : {} ;'.format(validate_loss, validate_top1))
            writer.add_scalars('Quantization-Loss/Training-Validate',{
                'train_loss': train_loss,
                'validate_loss': validate_loss
            }, epoch + 1)
            writer.add_scalars('Quantization-Top1/Training-Validate',{
                'train_acc1': train_top1,
                'validate_acc1': validate_top1
            }, epoch + 1)
            writer.add_scalars('Learning-Rate-For-Finetune', {
                'basic optimizer': fin_optimizer.state_dict()['param_groups'][0]['lr'],
            }, epoch + 1)
            is_best = False
            if validate_top1 > best_acc1:
                best_acc1 = validate_top1
                is_best = True
                logging.info('the best model top1 is : {} and its epoch is {} !'.format(best_acc1, epoch))
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': fin_optimizer.state_dict(),
                'scheduler': fin_scheduler.state_dict(),
            }, is_best, save='logger/{}-{}-{}-models'.format('cifar', args.arch, args.quan_mode[-3:]))
            fin_scheduler.step()

if __name__ == '__main__':
    main()