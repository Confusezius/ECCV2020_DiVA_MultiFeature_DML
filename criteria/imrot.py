import torch, torch.nn as nn
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM       = True
REQUIRES_EMA_NETWORK = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()
        self.classifier    = torch.nn.Linear(opt.network_feature_dim, 4, bias=False).to(opt.device)
        self.lr            = opt.lr * 10
        self.name          = 'imrot'


    def forward(self, feature_batch, imrot_labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        pred_batch    = self.classifier(feature_batch)
        loss          = torch.nn.CrossEntropyLoss()(pred_batch, imrot_labels.to(pred_batch.device))
        return loss
