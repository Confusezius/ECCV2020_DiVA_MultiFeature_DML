"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
import random, numpy as np




"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.mixup_alpha  = opt.mixup_alpha
        self.mixup_no_manifold = opt.mixup_no_manifold

    def forward(self, x, labels=None):
        mixup_layer = random.randint(0,3) if labels is not None else -1
        if labels is not None and self.mixup_no_manifold: mixup_layer = 0
        
        if mixup_layer==0:
            x, labels = self.mixup(x, labels, self.mixup_alpha)

        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for i,layerblock in enumerate(self.layer_blocks):
            if mixup_layer==i+1: x, labels = self.mixup(x, labels, self.mixup_alpha)
            x = layerblock(x)

        x = self.model.avgpool(x)
        enc_out = x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)

        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)

        if labels is None:
            return x, enc_out
        else:
            return x, enc_out, labels



    def mixup(self, batch, labels, mixup_alpha=2.):
        onehot_labels = torch.FloatTensor(batch.shape[0], self.pars.n_classes)
        onehot_labels.zero_()
        onehot_labels.scatter_(1, labels.unsqueeze(1).detach().cpu().to(torch.long),1)

        lam        = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha>0 else 1.
        perm_index = np.random.permutation(batch.shape[0])

        batch      = lam*batch  + (1-lam)*batch[perm_index,:]
        labels     = lam*onehot_labels + (1-lam)*onehot_labels[perm_index,:]

        return batch, labels
