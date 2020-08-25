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

        self.name             = 'fid_batchmatch_sampler'
        self.requires_storage = True

        self.bigbs           = opt.data_batchmatch_bigbs
        self.update_storage  = not opt.data_storage_no_update
        self.num_batch_comps = opt.data_batchmatch_ncomps
        self.low_proj_dim    = opt.data_sampler_lowproj_dim

        self.n_jobs = 16

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
        import time

        ### Random Subset from Random classes
        bigb_idxs = np.random.choice(len(self.storage), self.bigbs, replace=True)
        bigbatch  = self.storage[bigb_idxs]

        print('Precomputing Indices... ', end='')
        start = time.time()
        def batchfinder(n_calls, pos):
            idx_sets         = self.fid_match(n_calls, pos)
            structured_batches = [list(bigb_idxs[idx_set]) for idx_set in idx_sets]
            # structured_batch = list(bigb_idxs[self.fid_match(bigbatch, batch_size=self.batch_size//self.samples_per_class)])
            #Add random per-class fillers to ensure that the batch is build up correctly.
            for i in range(len(structured_batches)):
                class_idxs = [self.image_list[idx][-1] for idx in structured_batches[i]]
                for class_idx in class_idxs:
                    structured_batches[i].extend([random.choice(self.image_dict[class_idx])[-1] for _ in range(self.samples_per_class-1)])

            return structured_batches

        n_calls            = int(np.ceil(self.sampler_length/self.n_jobs))
        self.epoch_indices = Parallel(n_jobs = self.n_jobs)(delayed(batchfinder)(n_calls, i) for i in range(self.n_jobs))
        self.epoch_indices = [x for y in self.epoch_indices for x in y]
        # self.epoch_indices = Parallel(n_jobs = self.n_jobs)(delayed(batchfinder)(self.storage[np.random.choice(len(self.storage), self.bigbs, replace=True)]) for _ in tqdm(range(self.sampler_length), desc='Precomputing Indices...'))

        print('Done in {0:3.4f}s.'.format(time.time()-start))

    def replace_storage_entries(self, embeddings, indices):
        self.storage[indices] = embeddings

    def create_storage(self, dataloader, model, device):
        with torch.no_grad():
            _ = model.eval()
            _ = model.to(device)

            embed_collect = []
            for i,input_tuple in enumerate(tqdm(dataloader, 'Creating data storage...')):
                embed = model(input_tuple[1].type(torch.FloatTensor).to(device))
                if isinstance(embed, tuple): embed = embed[0]
                embed = embed.cpu()
                embed_collect.append(embed)
            embed_collect = torch.cat(embed_collect, dim=0)
            self.storage = embed_collect


    def fid_match(self, calls, pos):
        """
        """
        coll = []
        for _ in range(calls):
        # for _ in tqdm(range(calls), desc='Finding Batches - Job {}'.format(pos+1), position=pos):
            bigbatch   = self.storage[np.random.choice(len(self.storage), self.bigbs, replace=False)]
            batch_size = self.batch_size//self.samples_per_class

            if self.low_proj_dim>0:
                low_dim_proj = nn.Linear(bigbatch.shape[-1],self.low_proj_dim,bias=False)
                with torch.no_grad(): bigbatch = low_dim_proj(bigbatch)
            bigbatch = bigbatch.numpy()

            bigbatch_mean = np.mean(bigbatch, axis=0).reshape(-1,1)
            bigbatch_cov  = np.cov(bigbatch.T)

            fid_collect, bigb_idxs = [], []

            for _ in range(self.num_batch_comps):
                subset_idxs = sorted(np.random.choice(len(bigbatch), batch_size, replace=False))
                bigb_idxs.append(subset_idxs)
                subset      = bigbatch[subset_idxs,:]

                subset_mean = np.mean(subset, axis=0).reshape(-1,1)
                subset_cov  = np.cov(subset.T)

                offset   = np.eye(subset_cov.shape[0])*1e-16
                cov_sqrt = linalg.sqrtm((bigbatch_cov+offset).dot(subset_cov+offset), disp=False)[0].real

                diff = bigbatch_mean-subset_mean
                fid  = diff.T.dot(diff) + np.trace(bigbatch_cov) + np.trace(subset_cov) - 2*np.trace(cov_sqrt)

                fid_collect.append(fid)

            coll.append(bigb_idxs[np.argmin(fid_collect)])

        return coll

    def __len__(self):
        return self.sampler_length