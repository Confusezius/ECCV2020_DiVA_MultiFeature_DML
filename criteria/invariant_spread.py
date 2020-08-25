import torch, torch.nn as nn
from tqdm import tqdm


"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_EMA_NETWORK = False

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

        self.temperature   = opt.diva_instdiscr_temperature
        self.name          = 'invariantspread'
        self.lr            = opt.lr
        self.reference_labels = torch.zeros(opt.bs//2).to(torch.long).to(opt.device)
        self.diag_mat         = 1 - torch.eye(opt.bs).to(opt.device)

    def forward(self, head_1, head_2):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        bs = len(head_1)
        # l_pos  = head_1.view(bs, 1, -1).bmm(head_2.view(bs, -1, 1)).squeeze(-1)
        # l_neg  = head_1.view(bs, -1).mm(torch.cat([head_1, head_2], dim=0).T)
        # logits = torch.cat([l_pos, l_neg], dim=1)
        # loss   = torch.nn.CrossEntropyLoss()(logits/self.temperature, self.reference_labels)

        #get positive innerproduct
        x = torch.cat([head_1, head_2], dim=0)
        reordered_x = torch.cat((x.narrow(0,bs,bs),x.narrow(0,0,bs)), 0)
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.detach()).sum(1).div_(self.temperature).exp_()

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().detach()).div_(self.temperature).exp_()*self.diag_mat
        all_div  = all_prob.sum(1)

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(x.shape[0],1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        loss = - (lnPmtsum + lnPonsum)/x.shape[0]



        return loss
