import torch, torch.nn as nn
from tqdm import tqdm
import numpy as np

"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
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

        self.pars = opt
        
        self.temperature   = opt.diva_moco_temperature
        self.momentum      = opt.diva_moco_momentum
        self.n_key_batches = opt.diva_moco_n_key_batches


        if opt.diva_moco_trainable_temp:
            self.temperature = torch.nn.Parameter(torch.tensor(self.temperature).to(torch.float))
        self.lr          = opt.diva_moco_temp_lr

        self.name  = 'fast_moco'
        self.reference_labels = torch.zeros(opt.bs).to(torch.long).to(opt.device)

        self.lower_cutoff = opt.diva_moco_lower_cutoff
        self.upper_cutoff = opt.diva_moco_upper_cutoff

    def update_memory_queue(self, embeddings):
        self.memory_queue = self.memory_queue[len(embeddings):,:]
        self.memory_queue = torch.cat([self.memory_queue, embeddings], dim=0)

    def create_memory_queue(self, model, dataloader, device, opt_key=None):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            self.memory_queue = []
            counter = 0
            load_count  = 0
            total_count = self.n_key_batches//len(dataloader) + int(self.n_key_batches%len(dataloader)!=0)
            while counter<self.n_key_batches-1:
                load_count += 1
                for i,input_tuple in enumerate(tqdm(dataloader, 'Filling memory queue [{}/{}]...'.format(load_count, total_count), total=len(dataloader))):
                    embed = model(input_tuple[1].type(torch.FloatTensor).to(device))
                    if isinstance(embed, tuple): embed = embed[0]

                    if opt_key is not None:
                        embed = embed[opt_key].cpu()
                    else:
                        embed = embed.cpu()

                    self.memory_queue.append(embed)

                    counter+=1
                    if counter>=self.n_key_batches:
                        break

            self.memory_queue = torch.cat(self.memory_queue, dim=0).to(device)

        self.n_keys = len(self.memory_queue)

    def shuffleBN(self, bs):
        forward_inds  = torch.randperm(bs).long().cuda()
        backward_inds = torch.zeros(bs).long().cuda()
        value = torch.arange(bs).long().cuda()
        backward_inds.index_copy_(0, forward_inds, value)
        return forward_inds, backward_inds


    def forward(self, query_batch, key_batch):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        bs  = len(query_batch)

        l_pos = query_batch.view(bs, 1, -1).bmm(key_batch.view(bs, -1, 1)).squeeze(-1)
        l_neg = query_batch.view(bs, -1).mm(self.memory_queue.T)

        ### Compute Distance Matrix
        bs,dim  = len(query_batch),query_batch.shape[-1]

        ab = torch.mm(query_batch, self.memory_queue.T).detach()
        a2 = torch.nn.CosineSimilarity()(query_batch, query_batch).unsqueeze(1).expand_as(ab).detach()
        b2 = torch.nn.CosineSimilarity()(self.memory_queue, self.memory_queue).unsqueeze(0).expand_as(ab).detach()
        #Euclidean Distances
        distance_weighting = (-2.*ab+a2+b2).clamp(min=0).sqrt()
        distances = (-2.*ab+a2+b2).clamp(min=0).sqrt().clamp(min=self.lower_cutoff)

        #Likelihood Weighting
        distance_weighting = ((2.0 - float(dim)) * torch.log(distances) - (float(dim-3) / 2) * torch.log(1.0 - 0.25 * (distances.pow(2))))
        if self.pars.diva_fixed:
            distance_weighting = torch.exp(distance_weighting - torch.max(distance_weighting, dim=1, keepdims=True).values)
        else:
            distance_weighting = torch.exp(distance_weighting - torch.max(distance_weighting))
        distance_weighting[distances>self.upper_cutoff] = 0
        distance_weighting = distance_weighting.clamp(min=1e-45)
        if self.pars.diva_fixed:
            distance_weighting = distance_weighting/torch.sum(distance_weighting, dim=1, keepdims=True)
        else:
            distance_weighting = distance_weighting/torch.sum(distance_weighting, dim=0)

        ###
        l_neg = l_neg*distance_weighting

        ### INCLUDE SHUFFLE BN
        logits = torch.cat([l_pos, l_neg], dim=1)

        if isinstance(self.temperature, torch.Tensor):
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature.clamp(min=1e-8, max=1e4), self.reference_labels)
        else:
            loss = torch.nn.CrossEntropyLoss()(logits/self.temperature, self.reference_labels)

        return loss
