import torch

"""================================================================================================="""
ALLOWED_MINING_OPS  = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True

### This implements the Signal-To-Noise Ratio Triplet Loss
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
        """
        super(Criterion, self).__init__()
        self.margin     = opt.loss_snr_margin
        self.reg_lambda = opt.loss_snr_reg_lambda
        self.beta       = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_snr_beta) if opt.loss_snr_beta else None
        self.batchminer = batchminer

        if self.batchminer.name=='distance': self.reg_lambda = 0

        self.name = 'snr'

        self.lr   = opt.loss_snr_beta_lr


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

        pos_snr  = torch.var(batch[anchors,:]-batch[positives,:], dim=1)/torch.var(batch[anchors,:], dim=1)
        neg_snr  = torch.var(batch[anchors,:]-batch[negatives,:], dim=1)/torch.var(batch[anchors,:], dim=1)

        reg_loss = torch.mean(torch.abs(torch.sum(batch[anchors,:],dim=1)))

        if self.beta is not None:
            pos_snr  = torch.nn.functional.relu(pos_snr-beta+self.margin)
            neg_snr  = torch.nn.functional.relu(beta-neg_snr+self.margin)
            snr_loss = torch.sum(pos_snr + neg_snr)
            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.))
            snr_loss = snr_loss/pair_count.clamp(min=1)
        else:
            snr_loss = torch.nn.functional.relu(pos_snr - neg_snr + self.margin)
            snr_loss = torch.sum(snr_loss)/torch.sum(snr_loss>0)

        loss = snr_loss + self.reg_lambda * reg_loss

        return loss
