import numpy as np, torch, torch.nn as nn, torch.nn.functional as F



"""================================================================================================="""
ALLOWED_MINING_OPS = ['npair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()

        self.tan_angular_margin = np.tan(np.pi/180*opt.loss_angular_alpha)
        self.lam            = opt.loss_angular_npair_ang_weight
        self.l2_weight      = opt.loss_angular_npair_l2
        self.batchminer     = batchminer

        self.name           = 'angular'


    def forward(self, batch, labels):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        ####NOTE: Normalize Angular Loss, but not normalize npair loss!
        anchors, positives, negatives = self.batchminer(batch, labels)
        anchors, positives, negatives = batch[anchors], batch[positives], batch[negatives]
        n_anchors, n_positives, n_negatives = F.normalize(anchors, dim=1), F.normalize(positives, dim=1), F.normalize(negatives, dim=-1)

        is_term1 = 4*self.tan_angular_margin**2*(n_anchors + n_positives)[:,None,:].bmm(n_negatives.permute(0,2,1))
        is_term2 = 2*(1+self.tan_angular_margin**2)*n_anchors[:,None,:].bmm(n_positives[:,None,:].permute(0,2,1))
        is_term1 = is_term1.view(is_term1.shape[0], is_term1.shape[-1])
        is_term2 = is_term2.view(-1, 1)

        inner_sum_ang = is_term1 - is_term2
        angular_loss = torch.mean(torch.log(torch.sum(torch.exp(inner_sum_ang), dim=1) + 1))


        inner_sum_npair = anchors[:,None,:].bmm((negatives - positives[:,None,:]).permute(0,2,1))
        inner_sum_npair = inner_sum_npair.view(inner_sum_npair.shape[0], inner_sum_npair.shape[-1])
        npair_loss = torch.mean(torch.log(torch.sum(torch.exp(inner_sum_npair), dim=1) + 1))

        loss = npair_loss + self.lam*angular_loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))

        return loss
