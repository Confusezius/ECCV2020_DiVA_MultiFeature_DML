from batchminer import distance, semihard, random, parametric, npair
from batchminer import random_distance, shared_full_distance, shared_neg_distance, intra_random
from batchminer import lifted, anticollapse_distance, anticollapse_semihard,anticollapse_cdistance

BATCHMINING_METHODS = {'random':random,
                       'semihard':semihard,
                       'distance':distance,
                       'anticollapse_distance':anticollapse_distance,
                       'anticollapse_cdistance':anticollapse_cdistance,
                       'npair':npair,
                       'parametric':parametric,
                       'lifted':lifted,
                       'random_distance': random_distance,
                       'intra_random': intra_random,
                       'shared_full_distance': shared_full_distance,
                       'shared_neg_distance':  shared_neg_distance,
                       'anticollapse_semihard':anticollapse_semihard}


def select(batchminername, opt):
    #####
    if batchminername not in BATCHMINING_METHODS: raise NotImplementedError('Batchmining {} not available!'.format(batchminername))

    batchmine_lib = BATCHMINING_METHODS[batchminername]

    return batchmine_lib.BatchMiner(opt)
