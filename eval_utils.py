import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from eval_lib.cython_eval import eval_market1501_wrap
from model import ft_net
import pdb
import numpy as np
import scipy.io
import time
import sys, os
import argparse

def load_network(network, model_name, which_epoch):
    save_path = os.path.join('./model',model_name,'net_%s.pth'%which_epoch)
    # network.load_state_dict(torch.load(save_path))
    network.load_state_dict(torch.load(save_path))
    return network

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        # if count%100 == 0:
        #     print(count)
        ff = torch.FloatTensor(n,2048).zero_()
        # add original and flipped features
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            _, outputs = model(input_img) 
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
    return features

def extract_feature_v1(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    new_label = []
    for data in dataloaders:
        img, label, _ = data
        n, c, h, w = img.size()
        count += n
        # if count%100 == 0:
        #     print(count)
        ff = torch.FloatTensor(n,2048).zero_()
        # add original and flipped features
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            logits, outputs = model(input_img) 
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features,ff), 0)
        _, tmp_label = logits.max(1)
        new_label = np.append(new_label,tmp_label)
        # predicted labels 
        
        
    return features, new_label

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_id_CUHK(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        camera = 2*(int(filename.split('_')[0])-1) + int(filename.split('_')[2])
        label = path.split('/')[-2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

def get_id_CUHK_v1(img_path):
    camera_id = []
    labels = []
    for path, v, _ in img_path:
        filename = path.split('/')[-1]
        camera = 2*(int(filename.split('_')[0])-1) + int(filename.split('_')[2])
        label = path.split('/')[-2]
        labels.append(int(label))
        camera_id.append(int(camera))
    return camera_id, labels

def get_id_v1(img_path):
    camera_id = []
    labels = []
    for path, v, _ in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_test_acc(model, image_datasets, dataloaders, use_gpu, max_rank=10):
    
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    try:
        g_camid,g_pid = get_id(gallery_path)
        q_camid,q_pid = get_id(query_path)
    except:
        g_camid,g_pid = get_id_CUHK(gallery_path)
        q_camid,q_pid = get_id_CUHK(query_path) 
        
    if use_gpu:
        model = model.cuda()
    # Extract feature
    g_feas = extract_feature(model,dataloaders['gallery'])
    # print(g_feas.shape)
    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(g_feas.size(0), g_feas.size(1)))
    q_feas = extract_feature(model,dataloaders['query'])
    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(q_feas.size(0), q_feas.size(1)))

    distmat = np.matmul(q_feas.data.numpy(),np.transpose(g_feas.data.numpy()))*(-1.0)
    CMC, mAP = eval_market1501_wrap(distmat, q_pid, g_pid, q_camid, g_camid, max_rank=10)
    return CMC, mAP


def extr_fea_train(model, image_datasets, dataloaders, use_gpu):
    
    gallery_path = image_datasets.imgs
    # pdb.set_trace()
    try:
        g_camid,g_pid = get_id_v1(gallery_path)
    except:
        g_camid,g_pid = get_id_CUHK_v1(gallery_path)
        
    if use_gpu:
        model = model.cuda()
    # Extract feature
    g_feas, pre_ids = extract_feature_v1(model, dataloaders)
    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(g_feas.size(0), g_feas.size(1)))

    return g_feas, pre_ids
    
def extr_fea_train_un(model, image_datasets, dataloaders, use_gpu):
    
    gallery_path = image_datasets.imgs
    # pdb.set_trace()
    # try:
        # g_camid,g_pid = get_id_v1(gallery_path)
    # except:
        # g_camid,g_pid = get_id_CUHK_v1(gallery_path)
        
    if use_gpu:
        model = model.cuda()
    # Extract feature
    g_feas = extract_feature_v1(model,dataloaders)
    # print("Extracted features for gallery set, obtained {}-by-{} matrix".format(g_feas.size(0), g_feas.size(1)))

    return g_feas


def extract_train_second_label(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    second_label = []
    for data in dataloaders:
        img, label, _ = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 2048).zero_()
        # add original and flipped features
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            logits, outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)

        # predicted second labels
        _, top2 = logits.topk(2,1)
        top2= top2.cpu()
        tmp_label = top2[:,1]
        for ii in range(len(label)):
            if not top2[ii,0]==label[ii]:
                tmp_label[ii] = top2[ii,0]
        second_label = np.append(second_label, tmp_label)

    return features, second_label
    
def extract_train_pred_label(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    pred_label = []
    for data in dataloaders:
        img, label, _ = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 2048).zero_()
        # add original and flipped features
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            logits, outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)

        # predicted second labels
        _, top2 = logits.topk(2,1)
        top2= top2.cpu()
        tmp_label = top2[:,0]
        for ii in range(len(label)):
            if not top2[ii,0]==label[ii]:
                tmp_label[ii] = top2[ii,0]
        pred_label = np.append(pred_label, tmp_label)

    return features, pred_label