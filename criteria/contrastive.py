import torch, torch.nn as nn, torch.nn.functional as F

"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
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
        self.pos_margin = opt.loss_contrastive_pos_margin
        self.neg_margin = opt.loss_contrastive_neg_margin
        self.batchminer = batchminer

        self.name           = 'contrastive'


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets = self.batchminer(batch, labels)

        anchors   = [triplet[0] for triplet in sampled_triplets]
        positives = [triplet[1] for triplet in sampled_triplets]
        negatives = [triplet[2] for triplet in sampled_triplets]

        pos_dists = torch.mean(F.relu(nn.PairwiseDistance(p=2)(batch[anchors,:], batch[positives,:]) -  self.pos_margin))
        neg_dists = torch.mean(F.relu(self.neg_margin - nn.PairwiseDistance(p=2)(batch[anchors,:], batch[negatives,:])))

        loss      = pos_dists + neg_dists

        return loss
