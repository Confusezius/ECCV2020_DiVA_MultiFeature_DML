"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections, itertools as it

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

### DML-specific Libraries
import parameters    as par
import utilities.misc as misc




"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
parser = par.wandb_parameters(parser)
parser = par.diva_parameters(parser)

##### Read in parameters
opt = parser.parse_args()





"""==================================================================================================="""
if opt.dataset=='online_products':
    opt.evaluation_metrics = ['e_recall@1', 'e_recall@10', 'e_recall@100', 'nmi', 'f1', 'mAP']

if 'shared' in opt.diva_features and 'selfsimilarity' in opt.diva_features and len(opt.diva_features)==3:
    opt.diva_decorrelations = ['selfsimilarity-discriminative', 'shared-discriminative', 'shared-selfsimilarity']
if 'shared' in opt.diva_features and len(opt.diva_features)==4:
    opt.diva_decorrelations = ['selfsimilarity-discriminative', 'shared-discriminative', 'intra-discriminative']
if 'dc' in opt.diva_features or 'imrot' in opt.diva_features:
    opt.diva_decorrelations = []
if 'all' in opt.evaltypes:
    """==== EVALUATE DIFFERENT EMBEDDING SPACE REWEIGHTINGS ===="""
    #Generally, there is a slight benefit in placing higher weights on non-discriminative features during testing.
    opt.evaltypes = []
    if len(opt.diva_features)==1:
        opt.evaltypes = copy.deepcopy(opt.diva_features)
    if len(opt.diva_features)==2:
        for comb in list(it.product(opt.evaltypes, opt.evaltypes)):
            comb_name   = 'Combined_'+comb[0]+'_'+comb[1]+'-1-1'
            comb_name_2 = 'Combined_'+comb[1]+'_'+comb[0]+'-1-1'
            if comb_name not in opt.evaltypes and comb_name_2 not in opt.evaltypes and comb[0]!=comb[1]:
                opt.evaltypes.append(comb_name)
        opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'-1-0.5')
        opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'-0.5-1')
    if len(opt.diva_features)==3:
        opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'-1.5-1-1')
        opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'-1-1-1')
        opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'-0.5-1-1')
    if len(opt.diva_features)==4:
        if opt.dataset!='online_products':
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.75-1.25-1.25-1.25')
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.5-1-1-1')
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.5-1.5-1.5-1.5')
        else:
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-1-1-1-1')
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-0.5-1-1-1')
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-1-0.5-0.5-0.5')
            opt.evaltypes.append('Combined_'+opt.diva_features[0]+'_'+opt.diva_features[1]+'_'+opt.diva_features[2]+'_'+opt.diva_features[3]+'-1.5-1-1-1')


"""==================================================================================================="""
### The following setting is useful when logging to wandb and running multiple seeds per setup:
### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
if opt.savename=='group_plus_seed':
    if opt.log_online:
        opt.savename = opt.group+'_s{}'.format(opt.seed)
    else:
        opt.savename = ''

### If wandb-logging is turned on, initialize the wandb-run here:
if opt.log_online:
    import wandb
    _ = os.system('wandb login {}'.format(opt.wandb_key))
    os.environ['WANDB_API_KEY'] = opt.wandb_key
    wandb.init(project=opt.project, group=opt.group, name=opt.savename, dir=opt.save_path)
    wandb.config.update(opt)



"""==================================================================================================="""
### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import metrics       as metrics
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger



"""==================================================================================================="""
full_training_start_time = time.time()



"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained



"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
# if not opt.use_data_parallel:
os.environ["CUDA_VISIBLE_DEVICES"]= str(opt.gpu[0])



"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)



"""==================================================================================================="""
##################### NETWORK SETUP ##################
#NOTE: Networks that can be used: 'bninception, resnet50, resnet101, alexnet...'
#>>>>  see import pretrainedmodels; pretrainedmodels.model_names
opt.device = torch.device('cuda')
mfeat_net = 'multifeature_resnet50' if 'resnet' in opt.arch else 'multifeature_bninception'
model      = archs.select(mfeat_net, opt)
opt.network_feature_dim = model.feature_dim

print('{} Setup for {} with {} batchmining on {} complete with #weights: {}'.format(opt.loss.upper(), opt.arch.upper(), opt.batch_mining.upper(), opt.dataset.upper(), misc.gimme_params(model)))

if opt.fc_lr<0:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
else:
    all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
    fc_params         = model.model.last_linear.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]

#####
selfsim_model = archs.select(mfeat_net, opt)
selfsim_model.load_state_dict(model.state_dict())

#####
_  = model.to(opt.device)
_  = selfsim_model.to(opt.device)




"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets    = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['evaluation']       = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['evaluation_train'] = torch.utils.data.DataLoader(datasets['evaluation_train'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']          = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)

train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
datasets['training'].include_aux_augmentations = True
dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)

opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)




"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)


"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion_dict = {}

for key in opt.diva_features:
    if 'discriminative' in key:
        criterion_dict[key], to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)

if len(opt.diva_decorrelations):
    criterion_dict['separation'],     to_optim  = criteria.select('adversarial_separation', opt, to_optim, None)
if 'selfsimilarity' in opt.diva_features:
    criterion_dict['selfsimilarity'], to_optim  = criteria.select(opt.diva_ssl, opt, to_optim, None)
if 'invariantspread' in opt.diva_features:
    criterion_dict['invariantspread'], to_optim = criteria.select('invariantspread', opt, to_optim, batchminer)


#############
if 'shared' in opt.diva_features:
    if opt.diva_sharing=='standard':
        shared_batchminer        = bmine.select('shared_neg_distance', opt)
        criterion_dict['shared'], to_optim = criteria.select(opt.loss, opt, to_optim, shared_batchminer)
    elif opt.diva_sharing=='random':
        random_shared_batchminer = bmine.select('random_distance', opt)
        criterion_dict['shared'], to_optim = criteria.select(opt.loss, opt, to_optim, random_shared_batchminer)
    elif opt.diva_sharing=='full':
        full_shared_batchminer   = bmine.select('shared_full_distance', opt)
        criterion_dict['shared'], to_optim = criteria.select(opt.loss, opt, to_optim, full_shared_batchminer)
    else:
        raise Exception('Sharing method {} not available!'.format(opt.diva_sharing))

#############
if 'intra' in opt.diva_features:
    if opt.diva_intra=='random':
        intra_batchminer = bmine.select('intra_random', opt)
    else:
        raise Exception('Intra-Feature method {} not available!'.format(opt.diva_intra))
    criterion_dict['intra'], to_optim = criteria.select(opt.loss, opt, to_optim, intra_batchminer)


#############
if 'dc' in opt.diva_features:
    criterion_dict['dc'], to_optim     = criteria.select('dc', opt, to_optim, batchminer)
if 'imrot' in opt.diva_features:
    criterion_dict['imrot'], to_optim  = criteria.select('imrot', opt, to_optim, batchminer)

for key in criterion_dict.keys():
    _ = criterion_dict[key].to(opt.device)

if 'selfsimilarity' in criterion_dict:
    criterion_dict['selfsimilarity'].create_memory_queue(selfsim_model, dataloaders['training'], opt.device, opt_key='selfsimilarity')
if 'imrot' in criterion_dict:
    dataloaders['training'].dataset.predict_rotations = True


"""============================================================================"""
#################### OPTIM SETUP ####################
optimizer    = torch.optim.Adam(to_optim)
scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


"""============================================================================"""
#################### METRIC COMPUTER ####################
metric_computer = metrics.MetricComputer(opt.evaluation_metrics)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
for epoch in range(opt.n_epochs):
    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    """======================================="""
    if train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()


    """======================================="""
    if 'dc' in criterion_dict and epoch%opt.diva_dc_update_f==0:
        criterion_dict['dc'].update_pseudo_labels(model, dataloaders['evaluation_train'], opt.device)

    """======================================="""
    ### Train one epoch
    start = time.time()
    _ = model.train()

    loss_collect = {'train':[], 'separation':[]}
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))

    for i,(class_labels, input, input_indices, aux_input, imrot_labels) in enumerate(data_iterator):
        ###################
        if 'invariantspread' in criterion_dict:
            input = torch.cat([input[:len(input)//2,:], aux_input[:len(input)//2]], dim=0)
        features  = model(input.to(opt.device))
        features, direct_features = features

        ###################
        if 'selfsimilarity' in criterion_dict:
            with torch.no_grad():
                ### Use shuffleBN to avoid information bleeding making samples interdependent.
                forward_shuffle, backward_reorder = criterion_dict['selfsimilarity'].shuffleBN(len(features['selfsimilarity']))
                selfsim_key_features              = selfsim_model(aux_input[forward_shuffle].to(opt.device))
                if isinstance(selfsim_key_features, tuple): selfsim_key_features = selfsim_key_features[0]
                selfsim_key_features              = selfsim_key_features['selfsimilarity'][backward_reorder]

        ###################
        loss = 0.
        for key, feature in features.items():
            if 'discriminative' in key:
                loss_discr = criterion_dict[key](feature, class_labels)
                loss = loss + loss_discr
        if 'selfsimilarity' in criterion_dict:
            loss_selfsim = criterion_dict['selfsimilarity'](features['selfsimilarity'], selfsim_key_features)
            loss = loss + opt.diva_alpha_ssl*loss_selfsim
        if 'shared' in features:
            loss_shared = criterion_dict['shared'](features['shared'], class_labels)
            loss = loss + opt.diva_alpha_shared*loss_shared
        if 'intra' in features:
            loss_intra = criterion_dict['intra'](features['intra'], class_labels)
            loss = loss + opt.diva_alpha_intra*loss_intra
        if 'invariantspread' in criterion_dict:
            head_1 = features['invariantspread'][:len(input)//2]
            head_2 = features['invariantspread'][len(input)//2:]
            loss_invsp = criterion_dict['invariantspread'](head_1, head_2)
            loss       = loss + loss_invsp
        if 'dc' in criterion_dict:
            loss_dc = criterion_dict['dc'](direct_features, input_indices)
            loss    = loss + loss_dc
        if 'imrot' in criterion_dict:
            loss_imrot = criterion_dict['imrot'](direct_features, imrot_labels)
            loss    = loss + loss_imrot
        if 'separation' in criterion_dict:
            loss_adv = criterion_dict['separation'](features)
            loss     = loss + loss_adv


        optimizer.zero_grad()
        loss.backward()


        ### Compute Model Gradients and log them!
        grads              = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
        if opt.diva_moco_trainable_temp:
            LOG.progress_saver['Train'].log('temp', criterion_dict['selfsimilarity'].temperature.cpu().detach().numpy(), group='Temp')

        ### Update network weights!
        optimizer.step()

        ###
        loss_collect['train'].append(loss.item())
        if 'separation' in criterion_dict:
            loss_collect['separation'].append(loss_adv.item())

        if 'selfsimilarity' in criterion_dict:
            ### Update Key Network
            for model_par, key_model_par in zip(model.parameters(), selfsim_model.parameters()):
                momentum = criterion_dict['selfsimilarity'].momentum
                key_model_par.data.copy_(key_model_par.data*momentum + model_par.data*(1-momentum))

            ###
            criterion_dict['selfsimilarity'].update_memory_queue(selfsim_key_features)

        ###
        iter_count += 1

        if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect['train'])))

        """======================================="""
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(features.detach().cpu(), input_indices)



    result_metrics = {'loss': np.mean(loss_collect['train'])}
    if 'separation' in criterion_dict:
        result_metrics['sep. loss'] = np.mean(loss_collect['separation'])

    ####
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))



    """======================================="""
    ### Evaluate -
    _ = model.eval()
    if opt.dataset in ['cars196', 'cub200', 'online_products']:
        test_dataloaders = [dataloaders['testing']]
    elif opt.dataset=='in-shop':
        test_dataloaders = [dataloaders['testing_query'], dataloaders['testing_gallery']]

    eval.evaluate(opt.dataset, LOG, metric_computer, test_dataloaders, model, opt, opt.evaltypes, opt.device)


    LOG.update(all=True)


    """======================================="""
    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('\n-----\n')




"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ''
full_training_time = time.time()-full_training_start_time
summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

summary_text += '---------------\n'
for sub_logger in LOG.sub_loggers:
    metrics       = LOG.graph_writer[sub_logger].ov_title
    summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

with open(opt.save_path+'/training_summary.txt','w') as summary_file:
    summary_file.write(summary_text)
