# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch implementation of **CoLaKG** (SIGIR'25): Comprehending Knowledge Graphs with Large Language Models for Recommender Systems. The method injects LLM-generated KG comprehension into recommendation models via semantic embeddings and a hybrid neighbor strategy (KG structure + semantic similarity).

## Pipeline

1. **LLM KG comprehension** (`llm_code/`): Call LLM API to get KG descriptions → embed with SimCSE → save `.pt` embeddings
2. **Training** (`rec_code/`): Load precomputed semantic embeddings + interaction data → train CoLaKG model

## Training commands

```bash
cd rec_code
sh train_movielens.sh   # MovieLens-1M
sh train_lastfm.sh      # LastFM
sh train_mind.sh        # MIND
```

Or directly: `CUDA_VISIBLE_DEVICES=0 python main.py --dataset=<ml-1m|lastfm|mind> --model=colakg ...`

Key flags (see [rec_code/parse.py](rec_code/parse.py) for full list):
- `--model`: `mf`, `lgn`, or `colakg`
- `--neighbor_k`: total neighbors per item (default 10)
- `--kg_neighbor_k` / `--sem_neighbor_k`: split between KG and semantic neighbors in hybrid mode
- `--recdim`: embedding dimension (default 64)
- `--layer`: LightGCN propagation layers (default 3)
- `--use_drop_edge` / `--keepprob`: edge dropout in LightGCN

## Architecture

### Entry point: [rec_code/main.py](rec_code/main.py)

1. Parses config via [rec_code/world.py](rec_code/world.py) (which calls [rec_code/parse.py](rec_code/parse.py))
2. Loads precomputed item/user semantic embeddings (`.pt` files)
3. Builds **hybrid neighbor matrix**: KG neighbors (from triplets, sampled by relation diversity) + semantic neighbors (from cosine similarity, excluding KG overlaps). Joint neighbor index: `[num_items, kg_k + sem_k]`
4. Builds **relation matrix** (`adj_relations`): real KG relation IDs for KG edges; similarity-bucketed pseudo-relations for semantic edges (4 buckets: ≥0.8, ≥0.6, ≥0.4, <0.4)
5. Instantiates model via register (model classes in [rec_code/model.py](rec_code/model.py), dataset in [rec_code/dataloader.py](rec_code/dataloader.py))
6. Runs BPR training loop with periodic testing (every 5 epochs) and stability score updates

### Models ([rec_code/model.py](rec_code/model.py))

- **PureMF**: Simple matrix factorization baseline
- **LightGCN**: Graph convolution baseline using user-item bipartite graph
- **CoLaKG**: Main model with these components:
  - LightGCN propagation on user-item graph → `users`, `items` embeddings
  - **Semantic fusion**: Linear projection of 1024-dim SimCSE embeddings → `latent_dim`, fused with LightGCN outputs via learned per-dimension sigmoid gate (0→1 with temperature)
  - **KG-aware aggregation** (`KGINAggregatorLite`): Multi-hop KG propagation with relation-aware neighbor attention (Q·K attention over neighbors), optional dynamic relation weights via MLP
  - **User intent modeling** (KGIN-style): `n_factors` learnable intent prototypes + disentangled attention weights, independence constraint via distance correlation or cosine similarity
  - **Stability scores**: Periodic EMA of item embedding norm (proxy for training stability), applied as scaling factor to KG-aggregated item representations
  - **KG warmup**: Linear warmup factor `(epoch/warmup_epochs)²` controls gradual introduction of KG/semantic signals

### Datasets ([rec_code/dataloader.py](rec_code/dataloader.py))

- `BasicDataset` → `Loader` (generic train.txt/test.txt format) for ml-1m, mind
- `LastFM` (legacy format with custom column files)
- All produce: `n_users`, `m_items`, `getSparseGraph()` (normalized Laplacian), `allPos`, `testDict`

### Data preprocessing ([data_preprocess/](data_preprocess/))

- `movie.py` / `lastfm-2k.ipynb` / `mind.ipynb`: Extract KG subgraphs, format LLM prompts
- `build_lastfm_duoguanxitu.py` / `build_lastfm_item_neighbors_from_kg.py`: KG neighbor construction

## Key dependencies

Python 3.8, PyTorch 1.13+cu117, torch-geometric 2.6.0, torch-scatter, sentence-transformers 3.0.1. Full list in [requirements.txt](requirements.txt).

## External data (Google Drive)

Some files exceed GitHub limits and must be downloaded separately (see README for links): semantic embeddings (`.pt`), LLM input/response JSONs, and original KG data.
