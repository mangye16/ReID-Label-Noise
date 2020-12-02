# -*- coding: UTF-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from tensorboardX import SummaryWriter

import sys
import json
import scipy
import os, time
import argparse
import numpy as np
import torchvision
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile
from model import ft_net
from eval_utils import get_test_acc, extract_train_second_label
from utils import *
import loader, loss

version = torch.__version__
# #####################################################################
# argsions
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu', default='0', type=str, help='gpu ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--seed', default=1, type=int, help='rng seed')
parser.add_argument('--model_dir', default='.checkpoint/', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/comp/mangye/dataset/', type=str, help='data dir')
parser.add_argument('--dataset', default='market', type=str, help='training data:Market1501, DukeMTMCreID, CUHK03')
parser.add_argument('--resume', default='', type=str, help='path of pretrained "model:./model/baseline/net_8.pth"')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--noise_ratio', default=0.2, type=float, help='percentage of noise data in the training')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--stage2', action='store_true', help='either use stage2')
parser.add_argument('--pattern', action='store_true', help='either use patterned noise')

args = parser.parse_args()

torch.manual_seed(args.seed)

# initialization
start_epoch = 0
weight_r = [0, 0]
best_acc = 0
test_epoch = 2
data_dir = args.data_dir + args.dataset

# suffix
if not args.pattern:
    suffix = args.dataset + '_core2_noise_{}_lr_{:1.1e}'.format(args.noise_ratio, args.lr)
else:
    suffix = args.dataset + '_core_Pnoise_{}_lr_{:1.1e}'.format(args.noise_ratio, args.lr)

print('model: ' + suffix)

# define the log path       
log_dir = './res/' + args.dataset + '_log/'
checkpoint_path = './res/checkpoint_' + args.dataset + '/'
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
writer = SummaryWriter(vis_log_dir)
sys.stdout = Logger(log_dir + suffix + '_os.txt')

# define the gpu id
str_ids = args.gpu.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print('using gpu: {}'.format(gpu_ids))

# #####################################################################
# Load Data
train_transform = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load training dataDatasetFolder
print('Starting loading training data: ', args.dataset)
train_dataset = loader.DatasetFolder(os.path.join(data_dir, 'train'), transform=train_transform)
class_names = train_dataset.classes
dataset_sizes_train = len(train_dataset)

use_gpu = torch.cuda.is_available()

# Define a model
model1 = ft_net(len(class_names))
model2 = ft_net(len(class_names))
if args.pattern:
    model_tmp = ft_net(len(class_names))
if use_gpu:
    model1 = model1.cuda()
    model2 = model2.cuda()
    if args.pattern:
        model_tmp = model_tmp.cuda()

# Load a pretrainied model
if args.resume:
    # two pretrained SELF models (ignore when you start)
    model_name1 = 'market_noise_0.2_batch_32_refine_lr_1.0e-02_model1.t'
    print('Initilizaing weights with ', args.resume)
    model_path1 = checkpoint_path + model_name1
    model1.load_state_dict(torch.load(model_path1))

    model_name2 = 'market_noise_0.2_batch_32_refine_lr_1.0e-02_model2.t'
    model_path2 = checkpoint_path + model_name2
    model2.load_state_dict(torch.load(model_path2))
else:
    print('Initilizaing weights with ImageNet')

# generate noisy label
if args.noise_ratio >= 0:
    if not args.pattern:
        print('adding random noisy label')
        trainLabels = torch.LongTensor([y for (p, y, w) in train_dataset.imgs])
        trainLabels_nsy, if_truelbl = gen_nosiy_lbl(trainLabels, args.noise_ratio, len(class_names))
    else:
        print('adding patterned noisy label')
        trainLabels = torch.LongTensor([y for (p, y, w) in train_dataset.imgs])

        model_path = './res/checkpoint/' + args.dataset + '_base_noise_0.0_lr_1.0e-02_epoch_60.t'
        model_tmp.load_state_dict(torch.load(model_path))

        tansform_bak = train_transform
        train_dataset.transform = test_transform
        temploader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8)

        model_tmp.eval()  # Set model to evaluate mode
        print('extract second label...')
        _, second_label = extract_train_second_label(model_tmp, temploader)
        del model_tmp
        trainLabels_nsy, if_truelbl = gen_pattern_nosiy_lbl(trainLabels, args.noise_ratio, len(class_names), second_label)


# generate instance weight    
if args.stage2:
    print('Generating sef-generated weights......')
    # TO DO
else:
    print('Setting same weights for all the instances...')
    for i in range(len(trainLabels_nsy)):
        train_dataset.imgs[i] = (train_dataset.imgs[i][0], trainLabels_nsy[i], 1)

    # load training dataDatasetFolder
dataloaders_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,
                                                num_workers=8)  # 8 workers may work faster

# load testing dataDatasetFolder
test_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), test_transform) for x in ['gallery', 'query']}
dataloaders_test = {
    x: torch.utils.data.DataLoader(test_dataset[x], batch_size=args.batchsize, shuffle=False, num_workers=8) for x in
    ['gallery', 'query']}

# self label refining loss
criterion1 = loss.LabelRefineLoss()
# label co-refining loss
criterion2 = loss.CoRefineLoss()

# optimizers for two networks
ignored_params1 = list(map(id, model1.model.fc.parameters())) + list(map(id, model1.classifier.parameters()))
base_params1 = filter(lambda p: id(p) not in ignored_params1, model1.parameters())
optimizer_1 = optim.SGD([
    {'params': base_params1, 'lr': args.lr},
    {'params': model1.model.fc.parameters(), 'lr': args.lr * 10},
    {'params': model1.classifier.parameters(), 'lr': args.lr * 10}
], weight_decay=5e-4, momentum=0.9, nesterov=True)
ignored_params2 = list(map(id, model2.model.fc.parameters())) + list(map(id, model2.classifier.parameters()))
base_params2 = filter(lambda p: id(p) not in ignored_params2, model2.parameters())
optimizer_2 = optim.SGD([
    {'params': base_params2, 'lr': args.lr},
    {'params': model2.model.fc.parameters(), 'lr': args.lr * 10},
    {'params': model2.classifier.parameters(), 'lr': args.lr * 10}
], weight_decay=5e-4, momentum=0.9, nesterov=True)


# Decay LR by a factor of 0.1 every 20 epochs
def adjust_learning_rate(optimizer_1, optimizer_2, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer_1.param_groups[0]['lr'] = lr
    for i in range(len(optimizer_1.param_groups) - 1):
        optimizer_1.param_groups[i + 1]['lr'] = lr * 10

    optimizer_2.param_groups[0]['lr'] = lr
    for i in range(len(optimizer_2.param_groups) - 1):
        optimizer_2.param_groups[i + 1]['lr'] = lr * 10


def save_network(network1, network2, epoch_label, is_best=False):
    if is_best:
        save_path1 = checkpoint_path + suffix + 'model1_epoch_best.t'
        save_path2 = checkpoint_path + suffix + 'model2_epoch_best.t'
    else:
        save_path1 = checkpoint_path + suffix + 'model1_epoch_{}.t'.format(epoch_label)
        save_path2 = checkpoint_path + suffix + 'model2_epoch_{}.t'.format(epoch_label)

    torch.save(network1.state_dict(), save_path1)
    torch.save(network2.state_dict(), save_path2)


def train_model(model1, model2, criterion1, criterion2, optimizer_1, optimizer_2, epoch, weight_r):
    adjust_learning_rate(optimizer_1, optimizer_2, epoch)
    train_loss1 = AverageMeter()
    train_loss2 = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    model1.train()
    model2.train()

    correct1 = 0
    correct2 = 0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets, weights) in enumerate(dataloaders_train):
        if use_gpu:
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
            weights = Variable(weights.cuda())
        data_time.update(time.time() - end)

        # model inputs
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        # optimization
        if epoch <= 20:
            # self refining in first stage for the first network
            loss1 = criterion1(outputs1, targets, weight_r[0])
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()

            # self refining in first stage for the second network
            loss2 = criterion1(outputs2, targets, weight_r[1])
            optimizer_2.zero_grad()
            loss2.backward()
            optimizer_2.step()
        else:
            # co-refining in second stage for the first network
            loss1 = criterion2(outputs1, outputs2.detach(), targets, 1)
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()

            # co-refining in second stage for the second network
            loss2 = criterion2(outputs2, outputs1.detach(), targets, 1)
            optimizer_2.zero_grad()
            loss2.backward()
            optimizer_2.step()

        # log loss
        train_loss1.update(loss1.item(), inputs.size(0))
        train_loss2.update(loss2.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy
        _, predicted1 = outputs1.max(1)
        correct1 += predicted1.eq(targets).sum().item()

        _, predicted2 = outputs2.max(1)
        correct2 += predicted2.eq(targets).sum().item()

        total += inputs.size(0)

        if batch_idx % 10 == 0:
            print('Epoch:[{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Loss1: {train_loss1.val:.4f} ({train_loss1.avg:.4f}) '
                  'Loss2: {train_loss2.val:.4f} ({train_loss2.avg:.4f}) '
                  'Acc1: {:.2f}  Acc2: {:.2f}'.format(
                   epoch, batch_idx, len(dataloaders_train), 100. * correct1 / total, 100. * correct2 / total,
                   batch_time=batch_time, train_loss1=train_loss1, train_loss2=train_loss2))

    writer.add_scalar('trainAcc', 100. * correct1 / total, epoch)
    writer.add_scalar('trainAcc2', 100. * correct2 / total, epoch)
    writer.add_scalar('loss1', train_loss1.avg, epoch)
    writer.add_scalar('loss2', train_loss2.avg, epoch)

    weight_r = [1. / (1. + train_loss1.avg), 1. / (1. + train_loss2.avg)]
    return weight_r


for epoch in range(start_epoch, start_epoch + 61):

    # training
    print('Start Training..........')
    weight_r = train_model(model1, model2, criterion1, criterion2, optimizer_1, optimizer_2, epoch, weight_r)

    if epoch % test_epoch == 0:
        model1.eval()  # Set model to evaluate mode
        start = time.time()
        cmc, mAP = get_test_acc(model1, test_dataset, dataloaders_test, use_gpu, max_rank=10)
        writer.add_scalar('rank1', cmc[0], epoch)
        writer.add_scalar('mAP', mAP, epoch)
        if cmc[0] > best_acc:
            best_epoch = epoch
            best_acc = cmc[0]
            save_network(model1, model2, epoch, is_best=True)
        print('Epoch {}: R1:{:.4%}   R5:{:.4%}  R10:{:.4%}  mAP:{:.4%} (Best Epoch[{}])'.format(
            epoch, cmc[0], cmc[4], cmc[9], mAP, best_epoch))
        print('Evaluation time: {}'.format(time.time() - start))

    # evaluation
    if epoch > 20 and epoch % 20 == 0:
        save_network(model1, model2, epoch, is_best=False)
