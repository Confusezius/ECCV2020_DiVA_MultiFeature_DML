import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import criteria

"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()

        self.pars = opt

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        ####
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        self.proxies            = torch.randn(self.num_proxies, self.embed_dim)/8.
        self.proxies            = torch.nn.Parameter(self.proxies)
        self.lr  = opt.lr * opt.loss_oproxy_lrmulti

        ###
        self.class_idxs         = torch.arange(self.num_proxies)

        self.name           = 'oproxy'

        pars = {'pos_alpha':opt.loss_oproxy_pos_alpha,
                'pos_delta':opt.loss_oproxy_pos_delta,
                'neg_alpha':opt.loss_oproxy_neg_alpha,
                'neg_delta':opt.loss_oproxy_neg_delta}

        self.pars = pars

        ###
        self.mode           = opt.loss_oproxy_mode
        self.euclidean      = opt.loss_oproxy_euclidean
        self.d_mode         = 'euclidean' if self.euclidean else 'cosine'

        ###
        self.f_soft = torch.nn.Softplus()


    def prep(self, thing):
        return 1.*torch.nn.functional.normalize(thing, dim=1)


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        ###
        bs          = len(batch)
        batch       = self.prep(batch)
        self.labels = labels.unsqueeze(1)

        ###
        self.u_labels    = self.labels.view(-1)
        self.same_labels = (self.labels.T == self.u_labels.view(-1,1)).to(batch.device).T
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(torch.float).to(batch.device).T

        ###
        if self.mode == "anchor":
            self.dim = 0
        elif self.mode == "nca":
            self.dim = 1

        ###
        loss = self.compute_proxyloss(batch)

        ###
        return loss

    ###
    def compute_proxyloss(self, batch):
        proxies     = self.prep(self.proxies)
        pars = {k:-p if self.euclidean and 'alpha' in k else p for k,p in self.pars.items()}
        ###
        pos_sims    = self.smat(batch, proxies[self.u_labels], mode=self.d_mode)
        sims        = self.smat(batch, proxies, mode=self.d_mode)
        ###
        w_pos_sims  = -pars['pos_alpha']*(pos_sims-pars['pos_delta'])
        w_neg_sims  =  pars['neg_alpha']*(sims-pars['neg_delta'])


        pos_s = self.masked_logsumexp(w_pos_sims,mask=self.same_labels,dim=self.dim,max=True if self.d_mode=='euclidean' else False)
        neg_s = self.masked_logsumexp(w_neg_sims,mask=self.diff_labels,dim=self.dim,max=False if self.d_mode=='euclidean' else True)

        pos_s  = self.f_soft(pos_s)
        neg_s  = self.f_soft(neg_s)

        pos_s, neg_s = pos_s.mean(), neg_s.mean()

        loss  = pos_s + neg_s

        return loss

    ###
    def smat(self, A, B, mode='cosine'):
        if mode=='cosine':
            return A.mm(B.T)
        elif mode=='euclidean':
            As, Bs = A.shape, B.shape
            return (A.mm(A.T).diag().unsqueeze(-1)+B.mm(B.T).diag().unsqueeze(0)-2*A.mm(B.T)).clamp(min=1e-20).sqrt()

    ###
    def masked_logsumexp(self, sims, dim=0, mask=None, max=True):
        if mask is None:
            return torch.logsumexp(sims, dim=dim)
        else:
            if not max:
                ref_v      = (sims*mask).min(dim=dim, keepdim=True)[0]
            else:
                ref_v      = (sims*mask).max(dim=dim, keepdim=True)[0]

            nz_entries = (sims*mask)
            nz_entries = nz_entries.max(dim=dim,keepdim=True)[0]+nz_entries.min(dim=dim,keepdim=True)[0]
            nz_entries = torch.where(nz_entries.view(-1))[0].view(-1)

            if not len(nz_entries):
                return torch.tensor(0).to(torch.float).to(sims.device)
            else:
                return torch.log((torch.sum(torch.exp(sims-ref_v.detach())*mask,dim=dim)).view(-1)[nz_entries])+ref_v.detach().view(-1)[nz_entries]

            # return torch.log((torch.sum(torch.exp(sims)*mask,dim=dim)).view(-1))[nz_entries]
