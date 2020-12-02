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
from test_eval_cython import get_test_acc, extr_fea_train
from utils import *
import loader, loss
import pdb

version =  torch.__version__
# #####################################################################
# argsions
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu',default='0', type=str,help='gpu ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--seed', default=1, type=int, help='rng seed')
parser.add_argument('--model_dir',default='.checkpoint/', type=str, help='output model name')
parser.add_argument('--data_dir',default='/home/comp/mangye/dataset/', type=str, help='data dir')
parser.add_argument('--dataset',default='duke',type=str, help='training data:Market1501, DukeMTMCreID')
parser.add_argument('--pretrained',default='',type=str, help='path of pretrained "model:./model/baseline/net_8.pth"')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--noise_ratio', default=0.2, type=float, help='percentage of noise data in the training')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=2, type=float, help='beta distribution: alpha')
parser.add_argument('--beta', default=6, type=float, help='beta distribution: beta')
parser.add_argument('--LabelWt', default=60, type=int, help='label refinment weight')
parser.add_argument('--weighttype', default=0, type=int, help='weight type: instance weight, class weight')
parser.add_argument('--stage2', action='store_true', help='training stage 2')

args = parser.parse_args()

torch.manual_seed(args.seed)

start_epoch  = 0
if args.stage2:
    start_epoch = start_epoch + 20
    
best_acc = 0
test_epoch = 2
lr = args.lr
data_dir = args.data_dir + args.dataset
suffix = args.dataset + '_noise_{}_'.format(args.noise_ratio)
if args.LabelWt > 0 or args.stage2:   
    suffix = suffix + 'batch_{}_wt_{}'.format(args.batchsize,args.LabelWt)    
else:
    suffix = suffix + 'batch_{}_baseline'.format(args.batchsize) 
    

if args.stage2:
    suffix = suffix + '_beta_{}_{}_lr_{:1.1e}'.format(args.alpha, args.beta, args.lr)
    suffix = suffix + '_w_st2_new'
else:
    suffix = suffix + '_lr_{:1.1e}'.format(args.lr)
    suffix = suffix + '_w_st1'
    
print ('model: ' + suffix)

# define the log path       
log_dir = './new_res/' + args.dataset + '_log/'
checkpoint_path = './res/checkpoint/' 
vis_log_dir = log_dir + suffix + '/'
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)  
test_log_file = open(log_dir + suffix + '.txt', "w")       
sys.stdout = Logger(log_dir  + suffix + '_os.txt')

# define the gpu id
str_ids = args.gpu.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

print ('using gpu: {}'.format(gpu_ids))

# #####################################################################
# Load Data
train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((288,144), interpolation=3),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
test_transform = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# load training dataDatasetFolder
print('Starting loading training data: ', args.dataset )
train_dataset = loader.DatasetFolder(os.path.join(data_dir, 'train'), transform=train_transform)
class_names = train_dataset.classes
dataset_sizes_train = len(train_dataset)

use_gpu = torch.cuda.is_available()

# Define a model
model = ft_net(len(class_names))

if use_gpu:
    model = model.cuda()
    
# Load a pretrainied model
if args.pretrained or args.stage2:
    # model_name = 'market_noise_0.2_batch_32_lambda_0.4_lr_1.0e-02_st1_epoch_best.t'
    model_name = '{}_noise_{}_batch_32_wt_60_lr_1.0e-02_w_st1_epoch_best.t'.format(args.dataset, args.noise_ratio)
    print('Initilizaing weights with {}'.format(model_name))
    model_path = checkpoint_path + model_name
    model.load_state_dict(torch.load(model_path))
else:
    print('Initilizaing weights with ImageNet')
    
# generate noisy label
if args.noise_ratio >= 0:
    trainLabels = torch.LongTensor([y for (p, y, w) in train_dataset.imgs])
    trainLabels_nsy, if_truelbl = gen_nosiy_lbl(trainLabels, args.noise_ratio, len(class_names))
    print('Finish adding noisy label')

# generate instance weight    
if args.stage2:
    print('Generating sef-generated weights......')
    weight_file = './new_res/' + 'new_{}_{}_weights.npy'.format(args.dataset, args.noise_ratio)
    label_file = './new_res/' + 'new_{}_{}_label.npy'.format(args.dataset, args.noise_ratio)
    # if os.path.exists(weight_file):
        # all_weights = np.load(weight_file)
        # pre_pids = np.load(label_file)
    # else:
    tansform_bak = train_transform
    train_dataset.transform = test_transform
    temploader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8)
    
    model.eval()  # Set model to evaluate mode
    print('Start extract features...')
    start = time.time()
    train_feas, pre_pids = extr_fea_train(model, train_dataset, temploader, use_gpu)
    
    print('Evaluation time: {}'.format(time.time()-start))
    indexs, ori_weight = gen_weights_dist(train_feas, trainLabels_nsy, class_names, args.alpha, args.beta)
    order = np.argsort(indexs)
    all_weights = ori_weight[order]
    np.save(weight_file, all_weights)
    np.save(label_file, pre_pids)
    train_dataset.transform = tansform_bak
    all_weights = all_weights.astype(np.float32)
    for i in range(len(trainLabels_nsy)):
        train_dataset.imgs[i] = (train_dataset.imgs[i][0], int(pre_pids[i]), all_weights[i])  
else:
    print('Setting same weights for all the instances...')
    for i in range(len(trainLabels_nsy)):
        train_dataset.imgs[i] = (train_dataset.imgs[i][0], trainLabels_nsy[i],1)        

    
dataloaders_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=8) # 8 workers may work faster

# load testing dataDatasetFolder
test_dataset = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,test_transform) for x in ['gallery','query']}
dataloaders_test = {x: torch.utils.data.DataLoader(test_dataset[x], batch_size=args.batchsize, shuffle=False, num_workers=8) for x in ['gallery','query']}

# Define loss functions
# if args.LabelWt>0:
    # criterion = loss.LabelRefineLoss(lambda1=args.LabelWt)
if args.stage2:
    criterion = loss.InstanceWeightLoss(weighted = 1)
else:
    criterion = nn.CrossEntropyLoss()

# optimizer
ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
         {'params': base_params, 'lr': lr},
         {'params': model.model.fc.parameters(), 'lr': lr*10},
         {'params': model.classifier.parameters(), 'lr': lr*10}
     ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

def save_network(network, epoch_label, is_best = False):
    if is_best:
        save_path = checkpoint_path + suffix + '_epoch_best.t'
    else:
        save_path = checkpoint_path + suffix + '_epoch_{}.t'.format(epoch_label)
    torch.save(network.state_dict(), save_path)
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        w = float(np.exp(-2.0 * phase * phase))
        return min(w,0.5)
    
def train_model(model, criterion, optimizer_ft, scheduler, epoch):
    
    scheduler.step()
    lambda1 = sigmoid_rampup(epoch, args.LabelWt)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    
    model.train()
    correct = 0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets, weights) in enumerate(dataloaders_train):
        if use_gpu:
            inputs = Variable(inputs.cuda())  
            targets = Variable(targets.cuda()) 
            weights = Variable(weights.cuda())
        data_time.update(time.time() - end)    
        
        optimizer_ft.zero_grad()
        
        outputs = model(inputs)
        
        if args.stage2:
            loss = criterion(outputs, targets, weights) 
        else:
            loss = criterion(outputs, targets, lambda1) 
        
        loss.backward()
        optimizer_ft.step()
        
        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
                 
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += inputs.size(0)
        
        if batch_idx%10==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                  epoch, batch_idx, len(dataloaders_train),100.*correct/total, batch_time=batch_time, data_time=data_time, train_loss=train_loss))

    writer.add_scalar('training acc (train)', 100.*correct/total, epoch)
    writer.add_scalar('loss',  train_loss.avg, epoch)


for epoch in range(start_epoch, start_epoch+41):

    # training
    print('Start Training..........')
    train_model(model, criterion, optimizer_ft, exp_lr_scheduler, epoch)
    
    # evaluation
    if epoch%test_epoch ==0:
        model.eval()  # Set model to evaluate mode
        start = time.time()
        cmc, mAP = get_test_acc(model, test_dataset, dataloaders_test, use_gpu, max_rank=10)
        if cmc[0] > best_acc:
            best_epoch = epoch
            best_acc = cmc[0]
            save_network(model, epoch, is_best = True)
        print('Epoch {}: R1:{:.4%}   R5:{:.4%}  R10:{:.4%}  mAP:{:.4%} (Best Epoch[{}])'.format(
            epoch, cmc[0],cmc[4],cmc[9], mAP ,best_epoch)) 
        print('Epoch {}: R1:{:.4%}   R5:{:.4%}  R10:{:.4%}  mAP:{:.4%} (Best Epoch[{}])'.format(
            epoch, cmc[0],cmc[4],cmc[9], mAP ,best_epoch), file = test_log_file) 
        test_log_file.flush()
        print('Evaluation time: {}'.format(time.time()-start))
        
    # if epoch%20==0:
        # save_network(model, epoch, is_best = False)