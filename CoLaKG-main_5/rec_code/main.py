# 顶层脚本
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import datetime
from os.path import join
import os
import register
# 移除：from register import dataset
from sklearn.metrics.pairwise import cosine_similarity

utils.set_seed(world.seed)
print(">>SEED:", world.seed)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
k = world.config['neighbor_k']

log_file = f"../logs/{world.dataset}_{world.model_name}_neighbor{str(k)}_{current_time}.txt"

item_semantic_emb = torch.load(world.item_semantic_emb_file)
user_semantic_emb = torch.load(world.user_semantic_emb_file)

num_items = register.dataset.m_items

# ------------------------------------------------------------------
# 混合邻居策略：KG 邻居 + 语义邻居
# KG 邻居保证关系覆盖，语义邻居保证相关性
# ------------------------------------------------------------------
kg_k = world.config['kg_neighbor_k']
sem_k = world.config['sem_neighbor_k']
triplets_file = os.path.join("..", "data", world.dataset, "triplets.txt")

# 1. 从 triplets 构建 KG 邻居（按关系类型多样性采样）
kg_neighbors = None
relation_dict = {}
max_r = 0

if os.path.exists(triplets_file):
    print(f"Loading KG triplets from {triplets_file} for hybrid neighbor strategy...")
    head_to_tails = {}
    with open(triplets_file, 'r') as f:
        for line in f:
            try:
                h, r, t = map(int, line.strip().split('\t'))
                relation_dict[(h, t)] = r
                max_r = max(max_r, r)
                if h < num_items and t < num_items:
                    if h not in head_to_tails:
                        head_to_tails[h] = {}
                    if r not in head_to_tails[h]:
                        head_to_tails[h][r] = []
                    head_to_tails[h][r].append(t)
            except ValueError:
                continue

    print(f"Found {len(relation_dict)} triplets, {max_r + 1} relation types, {len(head_to_tails)} heads.")

    # 预计算 KG 邻居文件路径
    kg_neighbors_file = os.path.join("..", "data", world.dataset, f"{world.dataset}_hybrid_kg_neighbors_k{kg_k}.pt")

    if os.path.exists(kg_neighbors_file):
        kg_neighbors = torch.load(kg_neighbors_file).numpy()
        print(f"Loaded precomputed KG neighbors from {kg_neighbors_file} shape={kg_neighbors.shape}")
    else:
        # 使用固定种子保证可复现
        rng = np.random.RandomState(world.seed)
        # 按关系类型多样性采样：每种关系取最多 ceil(kg_k / n_rel) 个 tail
        kg_neighbors = np.full((num_items, kg_k), -1, dtype=np.int64)
        for item_id in range(num_items):
            if item_id in head_to_tails:
                rel_groups = head_to_tails[item_id]
                n_rels = len(rel_groups)
                per_rel = max(1, (kg_k + n_rels - 1) // n_rels)
                sampled = []
                for r_id in sorted(rel_groups.keys()):
                    tails = list(rel_groups[r_id])
                    rng.shuffle(tails)
                    sampled.extend(tails[:per_rel])
                    if len(sampled) >= kg_k:
                        break
                sampled = sampled[:kg_k]
                for j, t in enumerate(sampled):
                    kg_neighbors[item_id, j] = t

        # 未填满的位置用自环填充
        for i in range(num_items):
            for j in range(kg_k):
                if kg_neighbors[i, j] < 0:
                    kg_neighbors[i, j] = i

        # 保存预计算结果
        torch.save(torch.from_numpy(kg_neighbors).long(), kg_neighbors_file)
        print(f"Saved precomputed KG neighbors to {kg_neighbors_file} shape={kg_neighbors.shape}")
else:
    print("Warning: triplets.txt not found. Falling back to pure semantic neighbors.")
    kg_k = 0
    sem_k = k

# 2. 构建语义邻居（排除与 KG 邻居重复的）+ 相似度分数
sem_neighbors = None
sem_sim_scores = None
if sem_k > 0:
    sem_neighbors_file = os.path.join("..", "data", world.dataset, f"{world.dataset}_hybrid_sem_neighbors_k{sem_k}_kg{kg_k}.pt")
    sem_sim_file = os.path.join("..", "data", world.dataset, f"{world.dataset}_hybrid_sem_sim_k{sem_k}_kg{kg_k}.pt")

    if os.path.exists(sem_neighbors_file) and os.path.exists(sem_sim_file):
        sem_neighbors = torch.load(sem_neighbors_file).numpy()
        sem_sim_scores = torch.load(sem_sim_file).numpy()
        print(f"Loaded precomputed semantic neighbors from {sem_neighbors_file} shape={sem_neighbors.shape}")
    else:
        cosine_sim_matrix = cosine_similarity(item_semantic_emb.numpy())
        sorted_indices = np.argsort(-cosine_sim_matrix, axis=1)

        sem_neighbors = np.full((num_items, sem_k), -1, dtype=np.int64)
        sem_sim_scores = np.zeros((num_items, sem_k), dtype=np.float32)
        for i in range(num_items):
            kg_set = set(kg_neighbors[i].tolist()) if kg_k > 0 else set()
            kg_set.add(i)
            picked = []
            picked_sims = []
            rank = 1
            while len(picked) < sem_k and rank < sorted_indices.shape[1]:
                candidate = sorted_indices[i, rank]
                if candidate not in kg_set:
                    picked.append(candidate)
                    picked_sims.append(cosine_sim_matrix[i, candidate])
                    kg_set.add(candidate)
                rank += 1
            for j, (t, s) in enumerate(zip(picked, picked_sims)):
                sem_neighbors[i, j] = t
                sem_sim_scores[i, j] = s

        for i in range(num_items):
            for j in range(sem_k):
                if sem_neighbors[i, j] < 0:
                    sem_neighbors[i, j] = i
                    sem_sim_scores[i, j] = 0.0

        torch.save(torch.from_numpy(sem_neighbors).long(), sem_neighbors_file)
        torch.save(torch.from_numpy(sem_sim_scores), sem_sim_file)
        print(f"Saved precomputed semantic neighbors to {sem_neighbors_file} shape={sem_neighbors.shape}")

# 3. 拼接混合邻居矩阵: [num_items, kg_k + sem_k]
if kg_k > 0:
    kg_neighbors = torch.from_numpy(kg_neighbors).long()
if sem_k > 0:
    sem_neighbors = torch.from_numpy(sem_neighbors).long()
if kg_k > 0 and sem_k > 0:
    adj_indices = torch.cat([kg_neighbors, sem_neighbors], dim=1)
    print(f"Hybrid neighbors: KG({kg_k}) + Semantic({sem_k}) = {kg_k + sem_k}")
elif kg_k > 0:
    adj_indices = kg_neighbors
    print(f"Pure KG neighbors: {kg_k}")
else:
    adj_indices = sem_neighbors
    print(f"Pure semantic neighbors: {sem_k}")

# 索引合法性清洗
num_items = register.dataset.m_items
L = adj_indices.size(1)
self_idx = torch.arange(num_items).unsqueeze(1).expand(num_items, L)
invalid = (adj_indices < 0) | (adj_indices >= num_items)
adj_indices = torch.where(invalid, self_idx, adj_indices).long()

# ------------------------------------------------------------------
# 构建关系矩阵 (adj_relations) 以支持 KGIN 关系感知聚合
# KG 邻居使用真实关系 ID
# 语义邻居按相似度分桶分配不同的关系 ID
# ------------------------------------------------------------------
adj_relations = None
n_relations = 1

if os.path.exists(triplets_file):
    n_relations = max_r + 1
    self_loop_relation = n_relations
    # 语义相似度分桶: 4 个桶对应 4 种语义关系
    # [0.8, 1.0) -> sem_rel_0 (高度相似)
    # [0.6, 0.8) -> sem_rel_1 (中度相似)
    # [0.4, 0.6) -> sem_rel_2 (低度相似)
    # [0.0, 0.4) -> sem_rel_3 (弱相似)
    n_sem_buckets = 4
    sem_bucket_bounds = [0.8, 0.6, 0.4, 0.0]
    sem_rel_start = n_relations + 1  # +1 for self-loop
    n_relations += 1 + n_sem_buckets  # self-loop + 4 semantic buckets

    print(f"Relations: {n_relations} (incl. self-loop={self_loop_relation}, semantic buckets={sem_rel_start}~{sem_rel_start + n_sem_buckets - 1})")

    adj_relations = torch.full_like(adj_indices, self_loop_relation)

    adj_np = adj_indices.cpu().numpy()
    rel_np = adj_relations.cpu().numpy()

    found_count = 0
    total_count = adj_np.size

    for i in range(adj_np.shape[0]):
        for j in range(adj_np.shape[1]):
            tail = adj_np[i, j]
            if i == tail:
                rel_np[i, j] = self_loop_relation
            elif j < kg_k and (i, tail) in relation_dict:
                rel_np[i, j] = relation_dict[(i, tail)]
                found_count += 1
            elif j >= kg_k and sem_sim_scores is not None:
                sim = sem_sim_scores[i, j - kg_k]
                bucket = n_sem_buckets - 1
                for b, threshold in enumerate(sem_bucket_bounds):
                    if sim >= threshold:
                        bucket = b
                        break
                rel_np[i, j] = sem_rel_start + bucket
            # else: KG 邻居但无匹配关系，保持 self_loop_relation

    adj_relations = torch.from_numpy(rel_np).to(world.device)
    print(f"Constructed adj_relations. Matched {found_count}/{total_count} edges with KG relations.")
else:
    print("Warning: triplets.txt not found. Using default relation matrix (all zeros).")
    adj_relations = torch.zeros_like(adj_indices).to(world.device)

Recmodel = register.MODELS[world.model_name](world.config, register.dataset, adj_indices, adj_relations, n_relations, item_semantic_emb, user_semantic_emb)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
    

with open(log_file, "w") as f:
    f.write("Training Log\n")
    f.write("====================\n")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()

        # 周期性更新稳定性分数（与 CoLaKG.update_stability_scores 配合）
        Recmodel.current_epoch = epoch + 1
        Recmodel.update_stability_scores()
        
        if epoch % 5 == 0:
            cprint("[TEST]")
            test_results = Procedure.Test(register.dataset, Recmodel, epoch, w, world.config['multicore'])
            log_message = f'TEST RESULTS at EPOCH[{epoch+1}/{world.TRAIN_epochs}]: {test_results}'
            print(log_message)
            with open(log_file, "a") as f:
                f.write(log_message + "\n")
        
        output_information = Procedure.BPR_train_original(register.dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        
        end = time.time()
        epoch_time = end - start
        
        log_message = f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information} - Time: {epoch_time:.2f} seconds'
        print(log_message)
        
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
        
        torch.save(Recmodel.state_dict(), weight_file)

finally:
    if world.tensorboard:
        w.close()
