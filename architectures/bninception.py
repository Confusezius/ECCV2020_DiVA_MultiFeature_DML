"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm





"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.name = opt.arch

        self.pool_base = F.avg_pool2d
        self.pool_aux  = F.max_pool2d if 'double' in opt.arch else None


    def forward(self, x):
        x = self.model.features(x)
        y = self.pool_base(x,kernel_size=x.shape[-1])
        if self.pool_aux is not None:
            y += self.pool_aux(x, kernel_size=x.shape[-1])
        x = y
        x = self.model.last_linear(x.view(len(x),-1))
        if not 'normalize' in self.name:
            return x
        return torch.nn.functional.normalize(x, dim=-1)
