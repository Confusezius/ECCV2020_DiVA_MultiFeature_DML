import argparse, os


#######################################
def basic_training_parameters(parser):
    parser.add_argument('--dataset',         default='cub200',   type=str,   help='Dataset to use.')
    parser.add_argument('--train_val_split', default=1,          type=float, help='Percentage with which the training dataset is split into training/validation.')


    ### General Training Parameters
    parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr',             default=-1,       type=float, help='Learning Rate for network parameters.')
    parser.add_argument('--n_epochs',          default=150,      type=int,   help='Number of training epochs.')
    parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
    parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
    parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
    parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
    parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
    parser.add_argument('--tau',               default=[10000],  nargs='+',type=int,help='Stepsize before reducing learning rate.')
    parser.add_argument('--use_sgd',           action='store_true',   help='Appendix to save folder name if any special information is to be included.')


    ##### Loss-specific Settings
    parser.add_argument('--loss',            default='margin',      type=str,   help='Choose between TripletLoss, ProxyNCA, ...')
    parser.add_argument('--batch_mining',    default='distance',    type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance, adaptive interval.')
    parser.add_argument('--extension',       default='none',        type=str,   help='Extension Method to standard metric learning losses')

    #####
    parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: dim=128 or 64 is used in most papers.')
    parser.add_argument('--arch',         default='resnet50_frozen_normalize',  type=str,   help='Underlying network architecture. Frozen denotes that \
                                                                                                  exisiting pretrained batchnorm layers are frozen, and normalize denotes normalization of the output embedding.')
    parser.add_argument('--not_pretrained',             action='store_true')

    #####
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@2', 'e_recall@4', 'nmi', 'f1', 'mAP_c'], type=str, help='Metrics to evaluate performance by.')
    parser.add_argument('--evaltypes',          nargs='+', default=['discriminative'], type=str, help='The network may produce multiple embeddings (ModuleDict). If the key is listed here, the entry will be evaluated on the evaluation metrics.\
                                                                                                       Note: One may use Combined_embed1_embed2_..._embedn-w1-w1-...-wn to compute evaluation metrics on weighted (normalized) combinations.')
    parser.add_argument('--storage_metrics',    nargs='+', default=['e_recall@1'], type=str, help='Improvement in these metrics on the testset trigger checkpointing.')
    parser.add_argument('--realistic_augmentation', action='store_true')
    parser.add_argument('--realistic_main_augmentation', action='store_true')

    ##### Setup Parameters
    parser.add_argument('--gpu',          default=[1], nargs='+', type=int,   help='Random seed for reproducibility.')
    parser.add_argument('--savename',     default='group_plus_seed',   type=str,   help='Appendix to save folder name if any special information is to be included.')
    parser.add_argument('--source_path',  default=os.getcwd()+'/../../Datasets',   type=str, help='Path to training data.')
    parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')

    return parser


#######################################
def diva_parameters(parser):
    ##### Multifeature Parameters
    parser.add_argument('--diva_ssl',                 default='fast_moco', type=str,       help='Random seed for reproducibility.')
    parser.add_argument('--diva_sharing',             default='random', type=str,          help='Random seed for reproducibility.')
    parser.add_argument('--diva_intra',               default='random', type=str,          help='Random seed for reproducibility.')
    parser.add_argument('--diva_features',            default=['discriminative'], nargs='+', type=str,   help='Random seed for reproducibility.')
    parser.add_argument('--diva_decorrelations',      default=[], nargs='+', type=str)
    parser.add_argument('--diva_rho_decorrelation',   default=[1500], nargs='+', type=float, help='Weights for adversarial Separation of embeddings.')

    ### Adversarial Separation Loss
    parser.add_argument('--diva_decorrnet_dim', default=512,     type=int, help='')
    parser.add_argument('--diva_decorrnet_lr',  default=0.00001, type=float, help='')

    ### Invariant Spread Loss
    parser.add_argument('--diva_instdiscr_temperature', default=0.1,   type=float, help='')

    ### Deep Clustering
    parser.add_argument('--diva_dc_update_f', default=2,    type=int, help='')
    parser.add_argument('--diva_dc_ncluster', default=300,  type=int, help='')

    ### (Fast) Momentum Contrast Loss
    parser.add_argument('--diva_moco_momentum',      default=0.9, type=float, help='')
    parser.add_argument('--diva_moco_temperature',   default=0.1, type=float, help='')
    parser.add_argument('--diva_moco_n_key_batches', default=50,  type=int, help='')
    parser.add_argument('--diva_moco_lower_cutoff',  default=0.5,  type=float, help='')
    parser.add_argument('--diva_moco_upper_cutoff',  default=1.4,  type=float, help='')

    parser.add_argument('--diva_moco_temp_lr',        default=0.0005,   type=float, help='')
    parser.add_argument('--diva_moco_trainable_temp', action='store_true', help='')

    ### Weights for each feature space training objective
    parser.add_argument('--diva_alpha_ssl',      default=0.3,  type=float, help='')
    parser.add_argument('--diva_alpha_shared',   default=0.3,  type=float, help='')
    parser.add_argument('--diva_alpha_intra',    default=0.3,  type=float, help='')

    return parser


#######################################
def wandb_parameters(parser):
    ### Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true')
    parser.add_argument('--wandb_key',       default='<your_api_key_here>',  type=str,   help='Options are currently: wandb & comet')
    parser.add_argument('--project',         default='DiVA_Sample_Runs',  type=str,   help='Appendix to save folder name if any special information is to be included.')
    parser.add_argument('--group',           default='Sample_Run',  type=str,   help='Appendix to save folder name if any special information is to be included.')

    return parser


#######################################
def loss_specific_parameters(parser):
    ### Contrastive Loss
    parser.add_argument('--loss_contrastive_pos_margin', default=0, type=float, help='positive and negative margins for contrastive pairs.')
    parser.add_argument('--loss_contrastive_neg_margin', default=1, type=float, help='positive and negative margins for contrastive pairs.')
    # parser.add_argument('--loss_contrastive_neg_margin', default=0.2, type=float, help='positive and negative margins for contrastive pairs.')

    ### Triplet-based Losses
    parser.add_argument('--loss_triplet_margin',       default=0.2,         type=float, help='Margin for Triplet Loss')

    ### MarginLoss
    parser.add_argument('--loss_margin_margin',       default=0.2,          type=float, help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta_lr',      default=0.0005,       type=float, help='Learning Rate for class margin parameters in MarginLoss')
    parser.add_argument('--loss_margin_beta',         default=1.2,          type=float, help='Initial Class Margin Parameter in Margin Loss')
    parser.add_argument('--loss_margin_nu',           default=0,            type=float, help='Regularisation value on betas in Margin Loss.')
    parser.add_argument('--loss_margin_beta_constant',                      action='store_true')

    ### ProxyNCA
    parser.add_argument('--loss_proxynca_lr',     default=0.0005,     type=float, help='Learning Rate for Proxies in ProxyNCALoss.')
    #NOTE: The number of proxies is determined by the number of data classes.

    ### NPair L2 Penalty
    parser.add_argument('--loss_npair_l2',     default=0.005,        type=float, help='L2 weight in NPair. Note: Set to 0.02 in paper, but multiplied with 0.25 in the implementation as well.')

    ### Angular Loss
    parser.add_argument('--loss_angular_alpha',             default=36, type=float, help='Angular margin in degrees.')
    parser.add_argument('--loss_angular_npair_ang_weight',  default=2,  type=float, help='relative weighting between angular and npair contribution.')
    parser.add_argument('--loss_angular_npair_l2',          default=0.005,  type=float, help='relative weighting between angular and npair contribution.')

    ### Multisimilary Loss
    parser.add_argument('--loss_multisimilarity_pos_weight', default=2,         type=float, help='Weighting on positive similarities.')
    parser.add_argument('--loss_multisimilarity_neg_weight', default=40,        type=float, help='Weighting on negative similarities.')
    parser.add_argument('--loss_multisimilarity_margin',     default=0.1,       type=float, help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_multisimilarity_thresh',     default=0.5,       type=float, help='Exponential thresholding.')

    ### Lifted Structure Loss
    parser.add_argument('--loss_lifted_neg_margin', default=1,     type=float, help='')
    parser.add_argument('--loss_lifted_l2',         default=0.005, type=float, help='')

    ### Binomial Deviance Loss
    parser.add_argument('--loss_binomial_pos_weight', default=2,         type=float, help='Weighting on positive similarities.')
    parser.add_argument('--loss_binomial_neg_weight', default=40,        type=float, help='Weighting on negative similarities.')
    parser.add_argument('--loss_binomial_margin',     default=0.1,       type=float, help='Distance margin for both positive and negative similarities.')
    parser.add_argument('--loss_binomial_thresh',     default=0.5,       type=float, help='Exponential thresholding.')

    ### Quadruplet Loss
    parser.add_argument('--loss_quadruplet_alpha1',  default=1,   type=float, help='')
    parser.add_argument('--loss_quadruplet_alpha2',  default=0.5, type=float, help='')

    ### Soft-Triplet Loss
    parser.add_argument('--loss_softtriplet_n_centroids',   default=10,       type=int,   help='')
    parser.add_argument('--loss_softtriplet_margin_delta',  default=0.01,    type=float, help='')
    parser.add_argument('--loss_softtriplet_gamma',         default=0.1,     type=float, help='')
    parser.add_argument('--loss_softtriplet_lambda',        default=20,      type=float, help='')
    parser.add_argument('--loss_softtriplet_reg_weight',    default=0.2,     type=float, help='')
    parser.add_argument('--loss_softtriplet_lr',            default=0.0005,  type=float, help='')

    ### Normalized Softmax Loss
    parser.add_argument('--loss_softmax_lr',           default=0.00001,   type=float, help='')
    parser.add_argument('--loss_softmax_temperature',  default=0.05,   type=float, help='')

    ### Histogram Loss
    parser.add_argument('--loss_histogram_nbins',  default=51, type=int, help='')

    ### SNR Triplet (with learnable margin) Loss
    parser.add_argument('--loss_snr_margin',      default=0.2,   type=float, help='')
    parser.add_argument('--loss_snr_reg_lambda',  default=0.005, type=float, help='')
    parser.add_argument('--loss_snr_beta',        default=0,     type=float, help='Example values: 0.2')
    parser.add_argument('--loss_snr_beta_lr',     default=0.0005,type=float, help='Example values: 0.2')

    ### Normalized Softmax Loss
    parser.add_argument('--loss_arcface_lr',             default=0.0005,  type=float, help='')
    parser.add_argument('--loss_arcface_angular_margin', default=0.5,     type=float, help='')
    parser.add_argument('--loss_arcface_feature_scale',  default=64,      type=float, help='')

    ### Quadruplet Loss
    parser.add_argument('--loss_quadruplet_margin_alpha_1',  default=0.2, type=float, help='')
    parser.add_argument('--loss_quadruplet_margin_alpha_2',  default=0.2, type=float, help='')

    return parser



#######################################
def batchmining_specific_parameters(parser):
    ### Distance-based_Sampling
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float)
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float)
    return parser


#######################################
def batch_creation_parameters(parser):
    parser.add_argument('--data_sampler',              default='class_random', type=str, help='How the batch is created.')
    parser.add_argument('--samples_per_class',         default=2,              type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_batchmatch_bigbs',     default=512,         type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_batchmatch_ncomps',    default=10,         type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_storage_no_update',    action='store_true', help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_d2_coreset_lambda',    default=1, type=float, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_gc_coreset_lim',       default=1e-9, type=float, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_sampler_lowproj_dim',  default=-1, type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_sim_measure',          default='euclidean', type=str, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_gc_softened',          action='store_true', help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_idx_full_prec',        action='store_true', help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_mb_mom',               default=-1, type=float, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
    parser.add_argument('--data_mb_lr',                default=1, type=float, help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')

    return parser
