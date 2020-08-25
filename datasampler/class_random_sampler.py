import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list=None):
        self.image_dict         = image_dict
        self.image_list         = image_list

        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'class_random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            for _ in range(self.batch_size//self.samples_per_class):
                class_key = random.choice(list(self.image_dict.keys()))
                subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            yield subset

    def __len__(self):
        return self.sampler_length
