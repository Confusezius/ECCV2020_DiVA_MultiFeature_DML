import torch, torch.nn as nn



"""================================================================================================="""
ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

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
        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.thresh     = opt.loss_multisimilarity_thresh

        self.name           = 'multisimilarity'

    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        similarity = batch.mm(batch.T)

        loss = []
        for i in range(len(batch)):
            pos_idxs       = labels==labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels!=labels[i]

            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            ### This part doesn't really work, especially when you dont have a lot of positives in the batch...
            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            loss.append(pos_term + neg_term)

        loss = torch.mean(torch.stack(loss))
        return loss
