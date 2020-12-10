import os
import sys
import time
import logging
import numpy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import models.cifar100 as models 

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        net = models.vgg16_bn()
    elif args.net == 'vgg13':
        net = models.vgg13_bn()
    elif args.net == 'vgg11':
        net = models.vgg11_bn()
    elif args.net == 'vgg19':
        net = models.vgg19_bn()
    elif args.net == 'densenet121':
        net = models.densenet121()
    elif args.net == 'densenet161':
        net = models.densenet161()
    elif args.net == 'densenet169':
        net = models.densenet169()
    elif args.net == 'densenet201':
        net = models.densenet201()
    elif args.net == 'googlenet':
        net = models.googlenet()
    elif args.net == 'inceptionv3':
        net = models.inceptionv3()
    elif args.net == 'inceptionv4':
        net = models.inceptionv4()
    elif args.net == 'inceptionresnetv2':
        net = models.inception_resnet_v2()
    elif args.net == 'xception':
        net = models.xception()
    elif args.net == 'resnet18':
        net = models.resnet18()
    elif args.net == 'resnet34':
        net = models.resnet34()
    elif args.net == 'resnet50':
        net = models.resnet50()
    elif args.net == 'resnet101':
        net = models.resnet101()
    elif args.net == 'resnet152':
        net = models.resnet152()
    elif args.net == 'preactresnet18':
        net = models.preactresnet18()
    elif args.net == 'preactresnet34':
        net = models.preactresnet34()
    elif args.net == 'preactresnet50':
        net = models.preactresnet50()
    elif args.net == 'preactresnet101':
        net = models.preactresnet101()
    elif args.net == 'preactresnet152':
        net = models.preactresnet152()
    elif args.net == 'resnext50':
        net = models.resnext50()
    elif args.net == 'resnext101':
        net = models.resnext101()
    elif args.net == 'resnext152':
        net = models.resnext152()
    elif args.net == 'shufflenet':
        net = models.shufflenet()
    elif args.net == 'shufflenetv2':
        net = models.shufflenetv2()
    elif args.net == 'squeezenet':
        net = models.squeezenet()
    elif args.net == 'mobilenet':
        net = models.mobilenet()
    elif args.net == 'mobilenetv2':
        net = models.mobilenetv2()
    elif args.net == 'nasnet':
        net = models.nasnet()
    elif args.net == 'attention56':
        net = models.attention56()
    elif args.net == 'attention92':
        net = models.attention92()
    elif args.net == 'seresnet18':
        net = models.seresnet18()
    elif args.net == 'seresnet34':
        net = models.seresnet34()
    elif args.net == 'seresnet50':
        net = models.seresnet50()
    elif args.net == 'seresnet101':
        net = models.seresnet101()
    elif args.net == 'seresnet152':
        net = models.seresnet152()
    elif args.net == 'wideresnet':
        net = models.wideresnet()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def train(net, train_loader, args, optimizer, epoch, writer, warmup_scheduler, loss_function):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(train_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        if batch_index % args.print_freq == 0:
            logging.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.batch_size + len(images),
                total_samples=len(train_loader.dataset)
            ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    logging.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(net, test_loader, args, writer, loss_function, epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        logging.info('GPU INFO.....')
        logging.info(torch.cuda.memory_summary())
    logging.info('Evaluating Network.....')
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s \n'.format(
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)