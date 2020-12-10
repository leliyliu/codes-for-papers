from numpy.lib.type_check import real
from tqdm import tqdm
import time
from warmup_scheduler import GradualWarmupScheduler

import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.network import accuracy, AverageMeter, cross_entropy_loss_with_soft_target

def add_weight_decay(model, weight_decay, skip_keys, grads):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        skip = True
        for no_grad in grads:
            if no_grad in name:
                skip = False
        if skip:
            continue
        added = False
        for skip_key in skip_keys: # skip_keys -> means no decay 
            if skip_key in name:
                no_decay.append(param)
                added = True
                break
        if not added:
            decay.append(param)
    
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def get_optimizer(model, args, no_grads=[]):
    print('define optimizer')
    params = add_weight_decay(model, weight_decay=args.weight_decay, skip_keys=['alpha', 'xmax', 'expand_','running_scale'], no_grads=no_grads)
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)

    return optimizer

def get_scheduler(optimizer, args):
    if args.lr_scheduler == 'CosineAnnealingLR':
        print('Use cosine scheduler')
        scheduler_next = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'StepLR':
        print('Use step scheduler, step size: {}, gamma: {}'.format(args.step_size, args.gamma))
        scheduler_next = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        print('Use MultiStepLR scheduler, milestones: {}, gamma: {}'.format(args.milestones, args.gamma))
        scheduler_next = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError
    if args.warmup_epoch <= 0:
        return scheduler_next
    print('Use warmup scheduler')
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=args.warmup_multiplier,
                                          total_epoch=args.warmup_epoch,
                                          after_scheduler=scheduler_next)
    return lr_scheduler

def initial_blocks(net, teacher_model, train_loader, optimizers, epoch):
        # switch to train mode
    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net).cuda()
    
    if not isinstance(teacher_model, nn.DataParallel):
        teacher_model = nn.DataParallel(teacher_model).cuda()

    # net = net.module
    # teacher_model = teacher_model.module

    net.train()
    nBatch = len(train_loader)

    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    with tqdm(total=nBatch,
                desc='Train Epoch #{}'.format(epoch + 1)) as t:
        end = time.time()
        for i, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            target = labels

            for optimizer in optimizers:
                optimizer.zero_grad()  
            # compute output
            fake_features, output = net(images)

            teacher_model.train()
            with torch.no_grad():
                real_features, _ = teacher_model(images)
                for i, feature in enumerate(real_features):
                    real_features[i] = feature.detach()

            loss = 0
            
            for fake_feature, real_feature in zip(fake_features, real_features):
                fake_feature, real_feature = fake_feature.reshape(fake_feature.shape[0], -1), real_feature.reshape(real_feature.shape[0], -1)
                logp_x = F.log_softmax(fake_feature, dim=1)
                p_y = F.softmax(real_feature, dim=1)
                kl_loss = F.kl_div(logp_x, p_y, reduction='batchmean')
                loss = loss + kl_loss

            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            
            t.set_postfix({
                'loss': losses.avg,
                'top1': top1.avg,
                'img_size': images.size(2),
                'data_time': data_time.avg,
            })
            t.update(1)
            end = time.time()
    return losses.avg, top1.avg

def finetune(net, teacher_model, args, train_loader, optimizer, epoch, train_criterion):
    # switch to train mode

    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net).cuda()
    
    if not isinstance(teacher_model, nn.DataParallel):
        teacher_model = nn.DataParallel(teacher_model).cuda()

    net = net.module
    teacher_model = teacher_model.module

    net.train()
    nBatch = len(train_loader)

    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    with tqdm(total=nBatch,
                desc='Train Epoch #{}'.format(epoch + 1)) as t:
        end = time.time()
        for i, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            # images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            target = labels

            optimizer.zero_grad()  
            # compute output
            _, output = net(images)
            loss = train_criterion(output, labels)

            teacher_model.train()
            with torch.no_grad():
                _, soft_logits = teacher_model(images)
                soft_logits = soft_logits.detach()
                soft_label = F.softmax(soft_logits, dim=1) 

            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)   

            loss = loss + args.kd_ratio * kd_loss

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            t.set_postfix({
                'loss': losses.avg,
                'top1': top1.avg,
                'img_size': images.size(2),
                'data_time': data_time.avg,
            })
            t.update(1)
            end = time.time()
    return losses.avg, top1.avg

def feature_train(net, teacher_model, args, run_config, optimizer, epoch, train_criterion):
    # switch to train mode
    cur_device = net.device 

    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net).cuda()
    
    if not isinstance(teacher_model, nn.DataParallel):
        teacher_model = nn.DataParallel(teacher_model).cuda()

    net = net.module
    teacher_model = teacher_model.module

    net.train()
    nBatch = len(run_config.train_loader)

    losses = AverageMeter()
    kl_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()

    with tqdm(total=nBatch,
                desc='Train Epoch #{}'.format(epoch + 1)) as t:
        end = time.time()
        for i, (images, labels) in enumerate(run_config.train_loader):
            data_time.update(time.time() - end)
            images, labels = images.to(cur_device), labels.to(cur_device)
            # images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            target = labels

            optimizer.zero_grad()  
            # compute output
            fake_feature, output = net(images)
            fake_feature = fake_feature[-1]
            loss = train_criterion(output, labels)

            teacher_model.train()
            with torch.no_grad():
                real_feature, soft_logits = teacher_model(images)
                real_feature = real_feature[-1]
                real_feature, soft_logits = real_feature.detach(), soft_logits.detach()
                soft_label = F.softmax(soft_logits, dim=1) 

            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)   
            
            fake_feature, real_feature = fake_feature.reshape(fake_feature.shape[0], -1), real_feature.reshape(real_feature.shape[0], -1)
            logp_x = F.log_softmax(fake_feature, dim=1)
            p_y = F.softmax(real_feature, dim=1)
            kl_loss = F.kl_div(logp_x, p_y, reduction='batchmean')

            loss = loss + args.kd_ratio * kd_loss + args.feature_ratio * kl_loss 

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
            kl_losses.update(kl_loss.item(), images.size(0))

            t.set_postfix({
                'loss': losses.avg,
                'top1': top1.avg,
                'top5': top5.avg,
                'img_size': images.size(2),
                'data_time': data_time.avg,
                'kl_loss': kl_losses.avg,
            })
            t.update(1)
            end = time.time()
    return losses.avg, top1.avg, top5.avg

def validate(net, test_criterion, epoch=0, run_str='', data_loader=None, no_logs=False, valid=False):
    if not isinstance(net, nn.DataParallel):
        net = nn.DataParallel(net).cuda()

    net = net.module 

    if valid:
        net.eval()
    else:
        net.train()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(data_loader),
                    desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to('cuda'), labels.to('cuda')
                # compute output
                _, output = net(images)
                loss = test_criterion(output, labels)
                # measure accuracy and record loss
                acc1 = accuracy(output, labels, topk=(1,))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'img_size': images.size(2),
                })
                t.update(1)
    return losses.avg, top1.avg 