import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
from scipy import linalg


"""======================================================"""
REQUIRES_STORAGE = True

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, opt, image_dict, image_list):
        self.image_dict = image_dict
        self.image_list = image_list

        self.batch_size         = opt.bs
        self.samples_per_class  = opt.samples_per_class
        self.sampler_length     = len(image_list)//opt.bs
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'firstmoment_batchmatch_sampler'
        self.requires_storage = True

        self.bigbs           = opt.data_batchmatch_bigbs
        self.update_storage  = not opt.data_storage_no_update
        self.num_batch_comps = opt.data_batchmatch_ncomps

        self.n_jobs = 16

        self.internal_image_dict = {self.image_list[i]:i for i in range(len(self.image_list))}


    def __iter__(self):
        for i in range(self.sampler_length):
            # ### Random Subset from Random classes
            # bigb_idxs = np.random.choice(len(self.storage), self.bigbs, replace=True)
            # bigbatch  = self.storage[bigb_idxs]
            #
            # structured_batch = list(bigb_idxs[self.fid_match(bigbatch, batch_size=self.batch_size//self.samples_per_class)])
            # #Add random per-class fillers to ensure that the batch is build up correctly.
            #
            # class_idxs = [self.image_list[idx][-1] for idx in structured_batch]
            # for class_idx in class_idxs:
            #     structured_batch.extend([random.choice(self.image_dict[class_idx])[-1] for _ in range(self.samples_per_class-1)])

            yield self.epoch_indices[i]



    def precompute_indices(self):
        from joblib import Parallel, delayed

        ### Random Subset from Random classes
        self.firstmoment_match()
        # self.epoch_indices = Parallel(n_jobs = self.n_jobs)(delayed(self.firstmoment_match)() for _ in tqdm(range(self.sampler_length), desc='Precomputing Indices...'))


    def replace_storage_entries(self, embeddings, indices):
        self.storage[indices] = embeddings

    def create_storage(self, dataloader, model, device):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            embed_collect = []
            for i,input_tuple in enumerate(tqdm(dataloader, 'Creating data storage...')):
                embed = model(input_tuple[1].type(torch.FloatTensor).to(device)).cpu()
                embed_collect.append(embed)
            embed_collect = torch.cat(embed_collect, dim=0)
            self.storage = embed_collect


    def spc_batchfinder(self, n_samples):
        ### SpC-Sample big batch:
        subset, classes = [], []
        ### Random Subset from Random classes
        for _ in range(n_samples//self.samples_per_class):
            class_key = random.choice(list(self.image_dict.keys()))
            # subset.extend([(class_key, random.choice(len(self.image_dict[class_key])) for _ in range(self.samples_per_class)])
            subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            classes.extend([class_key]*self.samples_per_class)
        return np.array(subset), np.array(classes)


    def get_distmat(self, arr):
        prod = np.matmul(arr, arr.T)
        sq   = prod.diagonal().reshape(arr.shape[0], 1)
        dist_matrix = np.sqrt(np.clip(-2*prod + sq + sq.T, 0.0001, None))
        return dist_matrix

    def firstmoment_match(self):
        """
        """
        bigb_data_idxs, bigb_data_classes = self.spc_batchfinder(self.bigbs)
        bigb_dict = {}
        for i, bigb_cls in enumerate(bigb_data_classes):
            if bigb_cls not in bigb_dict: bigb_dict[bigb_cls] = []
            bigb_dict[bigb_cls].append(i)

        bigbatch = self.storage[bigb_data_idxs].numpy()
        distmat  = self.get_distmat(bigbatch)
        # bigbatch_mean = np.mean(bigbatch, axis=0).reshape(-1,1)
        # bigbatch_cov  = np.cov(bigbatch.T)


        cost_collect, bigb_idxs = [], []

        for _ in range(self.num_batch_comps):
            subset_idxs = [np.random.choice(bigb_dict[np.random.choice(list(bigb_dict.keys()))], self.samples_per_class, replace=False) for _ in range(self.batch_size//self.samples_per_class)]
            subset_idxs = [x for y in subset_idxs for x in y]
            # subset_idxs = sorted(np.random.choice(len(bigbatch), batch_size, replace=False))
            bigb_idxs.append(subset_idxs)
            subset      = bigbatch[subset_idxs,:]
            distmat     = get_distmax(subset)

            from IPython import embed; embed()

            fid_collect.append(fid)

        bigb_ix      = bigb_idxs[np.argmin(fid_collect)]
        bigb_data_ix = bigb_data_idxs[bigb_ix]
        return bigb_data_ix

    def __len__(self):
        return self.sampler_length
