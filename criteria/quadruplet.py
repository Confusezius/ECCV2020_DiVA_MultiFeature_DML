import torch, numpy as np

"""================================================================================================="""
ALLOWED_MINING_OPS = ['random','semihard', 'distance', 'parametric', 'anticollapse_distance']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False


class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
        """
        super(Criterion, self).__init__()
        self.batchminer = batchminer

        self.name           = 'quadruplet'

        self.margin_alpha_1 = opt.loss_quadruplet_margin_alpha_1
        self.margin_alpha_2 = opt.loss_quadruplet_margin_alpha_2


    def triplet_distance(self, anchor, positive, negative):
        return torch.nn.functional.relu(torch.norm(anchor-positive, p=2, dim=-1)-torch.norm(anchor-negative, p=2, dim=-1)+self.margin_alpha_1)

    def quadruplet_distance(self, anchor, positive, negative, fourth_negative):
        return torch.nn.functional.relu(torch.norm(anchor-positive, p=2, dim=-1)-torch.norm(negative-fourth_negative, p=2, dim=-1)+self.margin_alpha_2)

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels:  nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets    = self.batchminer(batch, labels)

        anchors   = np.array([triplet[0] for triplet in sampled_triplets]).reshape(-1,1)
        positives = np.array([triplet[1] for triplet in sampled_triplets]).reshape(-1,1)
        negatives = np.array([triplet[2] for triplet in sampled_triplets]).reshape(-1,1)

        fourth_negatives = negatives!=negatives.T
        fourth_negatives = [np.random.choice(np.arange(len(batch))[idxs]) for idxs in fourth_negatives]

        triplet_loss     = self.triplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:])
        quadruplet_loss  = self.quadruplet_distance(batch[anchors,:],batch[positives,:],batch[negatives,:],batch[fourth_negatives,:])
        # triplet_loss     = torch.stack([self.triplet_distance(batch[anchor,:],batch[positive,:],batch[negative,:]) for anchor,positive,negative in zip(anchors, positives, negatives)])
        # quadruplet_loss  = torch.stack([self.quadruplet_distance(batch[anchor,:],batch[positive,:],batch[negative,:],batch[fourth_negative,:]) for anchor,positive,negative,fourth_negative in zip(anchors, positives, negatives, fourth_negatives)])

        return torch.mean(triplet_loss) + torch.mean(quadruplet_loss)
