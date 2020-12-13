# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.network import save_checkpoint
from exps.cifar100.run_tool import train, eval_training, get_network, get_training_dataloader, get_test_dataloader, WarmUpLR
from wrapper.qcode_wrapper import replace_conv_recursively
from utils.ptflops import get_model_complexity_info

if __name__ == '__main__':
    q_modes_choice = sorted(['kernel_wise', 'layer_wise'])
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net', type=str, default='mobilenetv2', help='net type')
    parser.add_argument('-g', '--gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b','--batch-size', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('--warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr','--learning-rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-j','--workers', type=int, default=4, help='the process number')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--epochs', default=200, type=int, help='training epochs')
    parser.add_argument('--milestones', default=[60, 120, 180], nargs='+', type=int,
                    help='milestones of MultiStepLR')
    parser.add_argument('--manual-seed', default=2, type=int, help='random seed is settled')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--log-dir', default='logger', type=str)
    parser.add_argument('--save', default='EXP', type=str, help='save for the tensor log')
    parser.add_argument('--save-model', type=str, default='cifar100-mobilenetv2-HSQ-models/model_best.pth.tar')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load pretrained model')
    parser.add_argument('--quan-mode', type=str, default='Conv2dLSQ', 
                    help='corresponding for the quantize conv')
    parser.add_argument('--q-mode', choices=q_modes_choice, default='layer_wise',
                    help='Quantization modes: ' + ' | '.join(q_modes_choice) +
                            ' (default: kernel-wise)')
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)  # 设置随机种子
    net = get_network(args)
    if args.pretrained:
        checkpoint = torch.load(os.path.join(args.log_dir, '{}-{}-models/model_best.pth.tar'.format('cifar100', args.net)))
        net.load_state_dict(checkpoint['net'])

    replace_conv_recursively(net, args.quan_mode, args)

    flops, params = get_model_complexity_info(net, (3,32,32), print_per_layer_stat=False)
    logging.info('the model after quantized flops is {} and its params is {} '.format(flops, params))

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #use tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    args.save = 'cifar100-{}-{}-{}-{}'.format(args.net, args.save, args.quan_mode[-3:], time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, args.save))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
    writer.add_graph(net, input_tensor)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join('{}/{}/log.txt'.format(args.log_dir, args.save)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)

    best_acc = 0.0
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile('{}/{}'.format(args.log_dir, args.save_model)):
            logging.info("=> loading checkpoint '{}/{}'".format(args.log_dir, args.save_model))
            checkpoint = torch.load(os.path.join(args.log_dir, args.save_model))
            logging.info('load best training file to test acc...')
            net.load_state_dict(checkpoint['net'])
            logging.info('best acc is {:0.2f}'.format(checkpoint['acc']))
            best_acc = checkpoint['acc']
            args.start_epoch = checkpoint['epoch']
        else:
            logging.info("=> no checkpoint found at '{}/{}'".format(args.log_dir, args.save_model))
            raise Exception('No such model saved !')

    for epoch in range(args.start_epoch, args.epochs):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(net, cifar100_training_loader, args, optimizer, epoch, writer, warmup_scheduler, loss_function)
        acc = eval_training(net, cifar100_test_loader, args, writer, loss_function, epoch)

        is_best = False
        if acc > best_acc:
            best_acc = acc
            logging.info('the best acc is {} in epoch {}'.format(best_acc, epoch))
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, save='logger/{}-{}-{}-models'.format('cifar100', args.net, args.quan_mode[-3:]))

    logging.info('the final best acc is {}'.format(best_acc))
    writer.close()
