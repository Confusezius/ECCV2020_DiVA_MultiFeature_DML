import torch, numpy as np

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

### This Implementation follows: https://github.com/azgo14/classification_metric_learning

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            margin:             Triplet Margin.
        """
        super(Criterion, self).__init__()
        self.par         = opt

        self.temperature = opt.loss_softmax_temperature

        self.class_map = torch.nn.Parameter(torch.Tensor(opt.n_classes, opt.embed_dim))
        stdv = 1. / np.sqrt(self.class_map.size(1))
        self.class_map.data.uniform_(-stdv, stdv)

        self.name           = 'softmax'

        self.lr = opt.loss_softmax_lr


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        class_mapped_batch = torch.nn.functional.linear(batch, torch.nn.functional.normalize(self.class_map, dim=1))

        loss = torch.nn.CrossEntropyLoss()(class_mapped_batch/self.temperature, labels.to(torch.long).to(self.par.device))

        return loss
