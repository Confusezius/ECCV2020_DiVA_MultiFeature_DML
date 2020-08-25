import torch, torch.nn as nn, torch.nn.functional as F

"""================================================================================================="""
ALLOWED_MINING_OPS = ['lifted']
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
        self.margin     = opt.loss_lifted_neg_margin
        self.l2_weight  = opt.loss_lifted_l2
        self.batchminer = batchminer

        self.name           = 'lifted'


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        anchors, positives, negatives = self.batchminer(batch, labels)

        loss = []
        for anchor, positive_set, negative_set in zip(anchors, positives, negatives):
            anchor, positive_set, negative_set = batch[anchor, :].view(1,-1), batch[positive_set, :].view(1,len(positive_set),-1), batch[negative_set, :].view(1,len(negative_set),-1)
            pos_term = torch.logsumexp(nn.PairwiseDistance(p=2)(anchor[:,:,None], positive_set.permute(0,2,1)), dim=1)
            neg_term = torch.logsumexp(self.margin - nn.PairwiseDistance(p=2)(anchor[:,:,None], negative_set.permute(0,2,1)), dim=1)
            loss.append(F.relu(pos_term + neg_term))

        loss = torch.mean(torch.stack(loss)) + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))
        return loss
