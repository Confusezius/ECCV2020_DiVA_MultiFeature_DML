import torch

"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance', 'anticollapse_cdistance']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### Standard Triplet Loss, finds triplets in Mini-batches.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
        """
        super(Criterion, self).__init__()
        self.margin     = opt.loss_triplet_margin
        self.batchminer = batchminer

        self.name           = 'triplet'


    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu((anchor-positive).pow(2).sum()-(anchor-negative).pow(2).sum()+self.margin)

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets = self.batchminer(batch, labels)
        loss             = torch.stack([self.triplet_distance(batch[triplet[0],:],batch[triplet[1],:],batch[triplet[2],:]) for triplet in sampled_triplets])

        return torch.mean(loss)
