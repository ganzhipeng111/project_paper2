#!/bin/bash

decay=1e-3
lr=0.001
layer=3
seed=2020
dataset="lastfm"
topks="[10,20]"
recdim=64
use_drop_edge=1
keepprob=0.8
batch_size=512
dropout_i=0.3
dropout_u=0.2
dropout_n=0.3
neighbor_k=10
kg_neighbor_k=6
sem_neighbor_k=4
item_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg.pt'
user_semantic_emb_file='../data/lastfm/lastfm_embeddings_simcse_kg_user.pt'
dynamic_intent_lambda=0.05 
kg_stability_lambda=0.02 
consistency_update_interval=20 
stability_alpha=0.1 
stability_start_epoch=40
kg_n_hops=2
kg_hop_decay=0.7
kg_hop_residual=1 
kg_warmup_epochs=30
gate_temperature=1.2
n_factors=4
sim_regularity=1e-4

CUDA_VISIBLE_DEVICES=0 python main.py --bpr_batch=$batch_size --decay=$decay --lr=$lr --layer=$layer --seed=$seed --dataset=$dataset --topks=$topks --recdim=$recdim --use_drop_edge=$use_drop_edge --keepprob=$keepprob --neighbor_k=$neighbor_k --kg_neighbor_k=$kg_neighbor_k --sem_neighbor_k=$sem_neighbor_k --dropout_i=$dropout_i --dropout_u=$dropout_u --dropout_n=$dropout_n --dynamic_intent_lambda=$dynamic_intent_lambda --kg_stability_lambda=$kg_stability_lambda --consistency_update_interval=$consistency_update_interval --stability_alpha=$stability_alpha --stability_start_epoch=$stability_start_epoch --item_semantic_emb_file=$item_semantic_emb_file --user_semantic_emb_file=$user_semantic_emb_file --kg_n_hops=$kg_n_hops --kg_hop_decay=$kg_hop_decay --kg_hop_residual=$kg_hop_residual --kg_warmup_epochs=$kg_warmup_epochs --gate_temperature=$gate_temperature --n_factors=$n_factors --sim_regularity=$sim_regularity