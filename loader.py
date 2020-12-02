import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os,sys
# import matplotlib.pyplot as plt

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(data_dir, class_to_idx):
    images = []
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(data_dir, target)
        instan_num = len(os.listdir(d))
        weight0 = 1.0/instan_num
        # weight0 = 1.0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if os.path.exists(os.path.join(root, fname)):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target],weight0)
                    images.append(item)
    return images

class DatasetFolder(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
        self.root = root
        self.loader = loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = samples
        self.targets = [s[1] for s in samples]
        class_sample_counts = np.unique([s[1] for s in samples], return_counts=True)[1]
        # weight_dem = sum(1. / class_sample_counts)
        weight_dem = 1
        self.weights = [s[2]/weight_dem for s in samples] 
        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target, weigit = self.imgs[index]
        # weigit = self.weights[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, weigit

    def __len__(self):
        return len(self.imgs)

