import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['lastfm', 'ml-1m', 'mind', 'fund']
all_models  = ['mf', 'lgn', 'colakg']

config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['use_drop_edge'] = args.use_drop_edge
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['neighbor_k'] = args.neighbor_k
config['kg_neighbor_k'] = args.kg_neighbor_k
config['sem_neighbor_k'] = args.sem_neighbor_k
config['dropout_i'] = args.dropout_i
config['dropout_u'] = args.dropout_u
config['dropout_n'] = args.dropout_n
config['dynamic_intent_lambda'] = args.dynamic_intent_lambda
# 映射 intent 温度，便于控制动态权重的柔和程度
config['intent_temp'] = args.intent_temp
config['kg_stability_lambda'] = args.kg_stability_lambda
config['consistency_update_interval'] = args.consistency_update_interval
config['stability_alpha'] = args.stability_alpha
config['stability_start_epoch'] = args.stability_start_epoch
config['kg_n_hops'] = args.kg_n_hops
config['kg_hop_decay'] = args.kg_hop_decay
config['kg_hop_residual'] = args.kg_hop_residual
config['kg_warmup_epochs'] = args.kg_warmup_epochs
config['gate_temperature'] = args.gate_temperature
config['n_factors'] = args.n_factors
config['sim_regularity'] = args.sim_regularity

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

item_semantic_emb_file = args.item_semantic_emb_file
user_semantic_emb_file = args.user_semantic_emb_file


# TRAIN_epochs = args.epochs
TRAIN_epochs = 2000
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

