import os
import torch
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'lastfm')

triplets_path = os.path.join(DATA_DIR, 'triplets.txt')
output_path = os.path.join(DATA_DIR, 'lastfm_neighbors_kg.pt')

neighbor_k = 10

with open(os.path.join(DATA_DIR, 'item2entity.txt'), 'r') as f:
    num_items = sum(1 for _ in f)
print(f"[info] num_items from item2entity.txt: {num_items}")

co_counts = [defaultdict(int) for _ in range(num_items)]

with open(triplets_path, 'r', encoding='latin-1') as f:
    count = 0
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        h = int(parts[0])
        r = int(parts[1])
        t = int(parts[2])
        if 0 <= h < num_items and 0 <= t < num_items and h != t:
            co_counts[h][t] += 1
            co_counts[t][h] += 1
        count += 1
        if count % 100000 == 0:
            print(f"[progress] processed {count} triplets", flush=True)

print(f"[info] processed {count} triplets total", flush=True)

neighbors = []
for iid in range(num_items):
    counts = co_counts[iid]
    if not counts:
        neighbors.append([iid] * neighbor_k)
        continue
    sorted_neighbors = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    topk = [idx for idx, _ in sorted_neighbors[:neighbor_k]]
    if len(topk) < neighbor_k:
        topk += [iid] * (neighbor_k - len(topk))
    neighbors.append(topk)

neighbors_tensor = torch.tensor(neighbors, dtype=torch.long)
torch.save(neighbors_tensor, output_path)
print(f"[done] saved: {output_path} shape={tuple(neighbors_tensor.shape)}")

non_self = sum(1 for i in range(num_items) for j in range(neighbor_k) if neighbors[i][j] != i)
print(f"[stats] non-self-loop entries: {non_self}/{num_items * neighbor_k} ({non_self / (num_items * neighbor_k) * 100:.1f}%)")
