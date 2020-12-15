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
import copy 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.network import save_checkpoint
from exps.cifar100.run_tool import feature_train, kd_train, eval_training, get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, add_weight_decay
from wrapper.qcode_wrapper import replace_conv_recursively
from utils.ptflops import get_model_complexity_info

def main():
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
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--epochs', default=100, type=int, help='training epochs')
    parser.add_argument('--milestones', default=[30, 60, 90], nargs='+', type=int,
                    help='milestones of MultiStepLR')
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--manual-seed', default=2, type=int, help='random seed is settled')
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    parser.add_argument('--log-dir', default='logger', type=str)
    parser.add_argument('--save', default='EXP', type=str, help='save for the tensor log')
    parser.add_argument("--kd-ratio", type=float, default=0.5, help="learning from soft label distribution")
    parser.add_argument('--save-model', type=str, default='initial-cifar100-mobilenetv2-HSQ-models/model_best.pth.tar')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load pretrained model')
    parser.add_argument('--quan-mode', type=str, default='Conv2dHSQ', 
                    help='corresponding for the quantize conv')
    parser.add_argument('--q-mode', choices=q_modes_choice, default='layer_wise',
                    help='Quantization modes: ' + ' | '.join(q_modes_choice) +
                            ' (default: kernel-wise)')
    parser.add_argument('--initial-epochs', type=int, default=20, help='initial epochs for specific blocks')
    parser.add_argument('--initial-lr', default=2e-2, type=float, help='learning rate for initial blocks')
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)  # 设置随机种子

    args.save = 'cifar100-feature-{}-{}-{}-{}'.format(args.net, args.save, args.quan_mode[-3:], time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=os.path.join(
            args.log_dir, args.save))
    input_tensor = torch.Tensor(1, 3, 32, 32).cuda()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    fh = logging.FileHandler(os.path.join('{}/{}/log.txt'.format(args.log_dir, args.save)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    net = get_network(args)
    if args.pretrained:
        checkpoint = torch.load(os.path.join(args.log_dir, '{}-{}-models/model_best.pth.tar'.format('cifar100', args.net)))
        net.load_state_dict(checkpoint['net'])

    teacher_model = copy.deepcopy(net)

    replace_conv_recursively(net, args.quan_mode, args)

    if not args.quan_mode == 'Conv2dDPQ':
        flops, params = get_model_complexity_info(net, (3,32,32), print_per_layer_stat=False)
        logging.info('the model after quantized flops is {} and its params is {} '.format(flops, params))
        writer.add_graph(net, input_tensor)

    logging.info('args = %s', args)
    

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

    stage_blocks = []
    stage_blocks.append(['pre', 'stage1', 'stage2'])
    stage_blocks.append(['stage3', 'stage4'])
    stage_blocks.append(['stage5', 'stage6'])
    stage_blocks.append(['stage7', 'conv1'])


    optimizers, schedulers = [], []
    for block in stage_blocks:
        params = add_weight_decay(net, weight_decay=args.weight_decay, skip_keys=['expand_','running_scale'], grads=block)
        optimizer = optim.SGD(params, lr=args.initial_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.initial_epochs), last_epoch=-1)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    initial_criterion = nn.MSELoss(reduce=True, reduction='mean')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #use tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if args.evaluate:
        acc = eval_training(net, cifar100_test_loader, args, writer, loss_function, 0)
        logging.info('the acc for validate currently is {}'.format(acc))
        return 

    best_acc = 0.0
    args.initial_start_epoch, args.start_epoch = 0, 0
    if args.resume:
        if os.path.isfile('{}/{}'.format(args.log_dir, args.save_model)):
            logging.info("=> loading checkpoint '{}/{}'".format(args.log_dir, args.save_model))
            checkpoint = torch.load(os.path.join(args.log_dir, args.save_model))
            logging.info('load best training file to test acc...')
            net.load_state_dict(checkpoint['net'])
            logging.info('best acc is {:0.2f}'.format(checkpoint['acc']))
            best_acc = checkpoint['acc']
            if 'initial' in args.save_model:
                args.initial_start_epoch = checkpoint['epoch']
            else:
                args.initial_start_epoch = args.initial_epochs
                args.start_epoch = checkpoint['epoch']
        else:
            logging.info("=> no checkpoint found at '{}/{}'".format(args.log_dir, args.save_model))
            raise Exception('No such model saved !')

    for epoch in range(args.initial_start_epoch, args.initial_epochs):
        feature_train(net, teacher_model, cifar100_training_loader, args, optimizers, epoch, writer, initial_criterion)
        acc = eval_training(net, cifar100_test_loader, args, writer, loss_function, epoch, tb=False)
        for scheduler in schedulers:
            scheduler.step()

        is_best = False
        if acc > best_acc:
            best_acc = acc
            logging.info('the best acc is {} in epoch {}'.format(best_acc, epoch))
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'acc': best_acc,
        }, is_best, save='logger/initial-{}-{}-{}-models'.format('cifar100', args.net, args.quan_mode[-3:]))

    for epoch in range(args.start_epoch, args.epochs):
        kd_train(net, teacher_model, cifar100_training_loader, args, optimizer, epoch, writer, warmup_scheduler, loss_function)
        acc = eval_training(net, cifar100_test_loader, args, writer, loss_function, epoch)
        if epoch > args.warm:
            train_scheduler.step(epoch)

        is_best = False
        if acc > best_acc:
            best_acc = acc
            logging.info('the best acc is {} in epoch {}'.format(best_acc, epoch))
            is_best = True
            if args.quan_mode == 'Conv2dDPQ':
                flops, params = get_model_complexity_info(net, (3,32,32), print_per_layer_stat=False)
                logging.info('the model after quantized flops is {} and its params is {} '.format(flops, params))
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, save='logger/feature-{}-{}-{}-models'.format('cifar100', args.net, args.quan_mode[-3:]))

    if args.quan_mode == 'Conv2dDPQ':
        flops, params = get_model_complexity_info(net, (3,32,32), print_per_layer_stat=False)
        logging.info('the model after quantized flops is {} and its params is {} '.format(flops, params))
        writer.add_graph(net, input_tensor)

    logging.info('the final best acc is {}'.format(best_acc))
    writer.close()

if __name__ == '__main__':
    main()