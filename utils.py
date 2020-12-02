import torch
import torch.nn as nn
import torch.nn.init as init

import os
import os.path as osp
import sys
import time
import math
import random
import numpy as np

from scipy import stats
from scipy.spatial import distance
import pdb

def gen_nosiy_lbl(orig_lable, noise_ratio, class_number):
    random.seed(1)
    label = orig_lable.numpy()
    label_noisy = label.copy()
    for idx in range(class_number):
        idx_sub = np.argwhere(label==idx)[:,0]
        idx_nsy = np.array(random.sample(idx_sub.tolist(), int(np.round(len(idx_sub)*noise_ratio))))
        for i in idx_nsy:
            tar_lbl = random.sample((list(range(idx))+list(range(idx+1,class_number))),1)
            label_noisy[i] = tar_lbl[0]
    if_true = np.array((label==label_noisy)*1)
    return label_noisy, if_true

def gen_pattern_nosiy_lbl(orig_lable, noise_ratio, class_number, second_label):
    random.seed(1)
    label = orig_lable.numpy()
    label_noisy = label.copy()
    for idx in range(class_number):
        idx_sub = np.argwhere(label==idx)[:,0]
        idx_nsy = np.array(random.sample(idx_sub.tolist(), int(np.round(len(idx_sub)*noise_ratio))))
        for i in idx_nsy:
            label_noisy[i] = int(second_label[i])
    if_true = np.array((label==label_noisy)*1)
    return label_noisy, if_true



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, weights, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a,weights) + (1 - lam) * criterion(pred, y_b,weights)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std
    
def get_mfea_thres(features, alpha, beta):
    l2dsts = []
    mean_fea = np.mean(features, 0)
    for i in range(features.shape[0]):
        l2dst = distance.euclidean(features[i,:], mean_fea)
        l2dsts.append(l2dst)
    l2dsts = np.array(l2dsts)
    l2dst_max = np.max(l2dsts)
    l2dst_min = np.min(l2dsts)
    if l2dst_max-l2dst_min == 0:
        l2dsts_norm = (l2dsts-l2dst_min)
    else:
        l2dsts_norm = (l2dsts-l2dst_min)/(l2dst_max-l2dst_min)
    # l2dsts_norm_sort = np.sort(l2dsts_norm)
    # print(l2dsts_norm_sort)
    weights = stats.beta.pdf(l2dsts_norm, alpha, beta)
    return weights

def gen_weights_dist(features, trainLabels_nsy, class_names, alpha, beta):
    features = features.data.numpy()
    all_weights = np.array([])
    indexs = np.array([])
    for i in range(len(class_names)):
        f_idxs = np.where(trainLabels_nsy == i)[0]
        sele_feas = features[f_idxs, :]
        weights = get_mfea_thres(sele_feas, alpha, beta)
        all_weights = np.concatenate([all_weights, weights])
        indexs = np.concatenate([indexs, f_idxs])
    return indexs, all_weights   
    
class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()   

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise               
class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count