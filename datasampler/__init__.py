import datasampler.class_random_sampler
import datasampler.random_sampler
import datasampler.rrandom_sampler
import datasampler.fid_batchmatch_sampler
import datasampler.greedy_coreset_sampler
import datasampler.spc_fid_batchmatch_sampler
import datasampler.distmoment_batchmatch_sampler
import datasampler.disthist_batchmatch_sampler
import datasampler.criterion_batchmatch_sampler
import datasampler.d2_coreset_sampler
import datasampler.full_random_sampler
import datasampler.full_greedy_coreset_sampler


def select(sampler, opt, image_dict, image_list=None, **kwargs):
    if 'batchmatch' in sampler:
        if sampler=='fid_batchmatch':
            sampler_lib = fid_batchmatch_sampler
        elif sampler=='distmoment_batchmatch':
            sampler_lib = distmoment_batchmatch_sampler
        elif sampler=='disthist_batchmatch':
            sampler_lib = disthist_batchmatch_sampler
        elif sampler=='spc_fid_batchmatch':
            sampler_lib = spc_fid_batchmatch_sampler
        elif sampler=='criterion_batchmatch':
            sampler_lib = criterion_batchmatch_sampler
    if 'random' in sampler:
        sampler_lib = random_sampler
        if 'class' in sampler:
            sampler_lib = class_random_sampler
        if 'full' in sampler:
            sampler_lib = full_random_sampler
    if 'rrandom'==sampler:
        sampler_lib = rrandom_sampler    
    if 'coreset' in sampler:
        if 'greedy' in sampler:
            if 'full' in sampler:
                sampler_lib = full_greedy_coreset_sampler
            else:
                sampler_lib = greedy_coreset_sampler
        if 'd2' in sampler:
            sampler_lib = d2_coreset_sampler

    sampler = sampler_lib.Sampler(opt,image_dict=image_dict,image_list=image_list)

    return sampler
