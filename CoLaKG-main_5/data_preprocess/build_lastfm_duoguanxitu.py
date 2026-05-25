import os
import random
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'lastfm')

item_map_file = os.path.join(DATA_DIR, 'item_map.txt')
tags_file = os.path.join(DATA_DIR, 'tags.dat')
user_tagged_file = os.path.join(DATA_DIR, 'user_taggedartists.dat')

out_item2entity = os.path.join(DATA_DIR, 'item2entity.txt')
out_triplets = os.path.join(DATA_DIR, 'triplets.txt')

print(f"[build_lastfm_duoguanxitu] DATA_DIR = {DATA_DIR}", flush=True)
for path in (item_map_file, tags_file, user_tagged_file):
    if not os.path.exists(path):
        print(f"Missing input file: {path}", flush=True)
        raise FileNotFoundError(path)

if os.environ.get("DUOGUANXITU_DRY", "0") == "1":
    print("[dry-run] inputs ok, exiting before heavy processing", flush=True)
    raise SystemExit(0)

# ── 1. 读取 item_map.txt: artistID itemID ──
artist2item = {}
item2artist = {}
with open(item_map_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        artist_id = int(parts[0])
        item_id = int(parts[1])
        artist2item[artist_id] = item_id
        item2artist[item_id] = artist_id

num_items = max(item2artist.keys()) + 1
print(f"[info] item_map: {len(artist2item)} artists -> items, item_id range: 0~{num_items - 1}", flush=True)

# ── 2. 读取 tags.dat: tagID \t tagValue ──
tag_names = {}
with open(tags_file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        try:
            tag_id = int(parts[0])
        except ValueError:
            continue
        tag_names[tag_id] = parts[1]
print(f"[info] tags: {len(tag_names)} entries", flush=True)

# ── 3. tag 语义分类 ──
GENRE_KW = [
    'rock', 'metal', 'pop', 'jazz', 'blues', 'folk', 'country', 'punk',
    'rap', 'hip hop', 'hip-hop', 'electronic', 'dance', 'classical',
    'reggae', 'soul', 'funk', 'rnb', 'r&b', 'indie', 'alternative',
    'grunge', 'ska', 'swing', 'disco', 'house', 'techno', 'trance',
    'dubstep', 'ambient', 'industrial', 'gothic', 'shoegaze', 'dub',
    'emo', 'grindcore', 'hardcore', 'post-rock', 'post-punk', 'prog',
    'psychedelic', 'synth', 'lo-fi', 'lofi', 'trip hop', 'trip-hop',
    'drum', 'bass', 'idm', 'glitch', 'noise', 'experimental',
    'avant-garde', 'minimal', 'downtempo', 'chillout', 'lounge', 'ebm',
    'electropop', 'electro pop', 'synthpop', 'synth-pop', 'new wave',
    'garage', 'surf', 'drone', 'acoustic', 'instrumental', 'soundtrack',
    'ballad', 'opera', 'choral', 'baroque',
]
MOOD_KW = [
    'melancholy', 'melancholic', 'dark', 'happy', 'sad', 'upbeat',
    'chill', 'relaxing', 'energetic', 'mellow', 'atmospheric', 'ethereal',
    'dreamy', 'moody', 'aggressive', 'intense', 'calm', 'peaceful',
    'romantic', 'sentimental', 'dramatic', 'gloomy', 'uplifting',
    'nostalgic', 'brooding', 'bittersweet', 'haunting', 'serene',
    'soothing', 'spacy', 'raw', 'fierce', 'beautiful', 'epic',
    'amazing', 'lovely', 'addictive', 'catchy', 'groovy', 'trippy',
]
ERA_KW = [
    '80s', '90s', '70s', '60s', '50s', '2000s', '00s', '10s',
    'classic', 'old school', 'new school', 'retro', 'vintage',
    'oldies', 'modern', 'contemporary',
]
REGION_KW = [
    'german', 'deutsch', 'british', 'french', 'spanish', 'japanese',
    'korean', 'greek', 'italian', 'polish', 'american', 'swedish',
    'norwegian', 'finnish', 'brazilian', 'russian', 'australian',
    'canadian', 'mexican', 'latin', 'asian', 'israeli', 'angola',
    'portuguese', 'nordic', 'scandinavian', 'european', 'uk',
    'new york', 'california', 'belarus', 'african', 'indian',
    'chinese', 'arab', 'turkish', 'irish', 'scottish', 'dutch',
    'belgian', 'austrian', 'swiss', 'czech',
]

REL_GENRE = 0
REL_MOOD = 1
REL_ERA = 2
REL_REGION = 3
REL_OTHER = 4
N_RELATIONS = 5

tag2rel = {}
for tid, tname in tag_names.items():
    tl = tname.lower()
    if any(kw in tl for kw in GENRE_KW):
        tag2rel[tid] = REL_GENRE
    elif any(kw in tl for kw in MOOD_KW):
        tag2rel[tid] = REL_MOOD
    elif any(kw in tl for kw in ERA_KW):
        tag2rel[tid] = REL_ERA
    elif any(kw in tl for kw in REGION_KW):
        tag2rel[tid] = REL_REGION
    else:
        tag2rel[tid] = REL_OTHER

rel_names = {REL_GENRE: 'genre', REL_MOOD: 'mood', REL_ERA: 'era',
             REL_REGION: 'region', REL_OTHER: 'other'}
print(f"[info] relation types: {N_RELATIONS} ({', '.join(rel_names.values())})", flush=True)

# ── 4. 读取 user_taggedartists.dat: userID artistID tagID ──
artist2tags = defaultdict(set)
with open(user_tagged_file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            artist_id = int(parts[1])
            tag_id = int(parts[2])
        except ValueError:
            continue
        artist2tags[artist_id].add(tag_id)
print(f"[info] artist2tags: {len(artist2tags)} artists have tags", flush=True)

# ── 5. 写 item2entity.txt: item_id entity_id (1:1, entity_id == item_id) ──
with open(out_item2entity, 'w', encoding='utf-8') as f:
    for item_id in range(num_items):
        f.write(f"{item_id}\t{item_id}\n")
print(f"[info] wrote item2entity.txt: {num_items} lines", flush=True)

# ── 6. 构建 item-tag 映射 ──
item2tags = defaultdict(set)
for artist_id, tags in artist2tags.items():
    if artist_id not in artist2item:
        continue
    item_id = artist2item[artist_id]
    for tag_id in tags:
        if tag_id in tag2rel:
            item2tags[item_id].add(tag_id)
print(f"[info] items with tags: {len(item2tags)}", flush=True)

# ── 7. 构建 tag2items 映射 ──
tag2items = defaultdict(set)
for item_id, tags in item2tags.items():
    for tag_id in tags:
        tag2items[tag_id].add(item_id)

# ── 8. 构建 item-item triplets ──
# 策略: 通过共同 tag 建立 item-item 关系
# 同一 tag 下的 item 互连，关系类型由 tag 语义决定
# 同一 (head, tail) 对只保留优先级最高的关系 (genre > mood > era > region > other)
# 限制每个 tag 下的 item 数量，避免热门 tag 产生过多 triplet

REL_PRIORITY = {REL_GENRE: 0, REL_MOOD: 1, REL_ERA: 2, REL_REGION: 3, REL_OTHER: 4}

MAX_ITEMS_PER_TAG = 50
pair2rel = {}

for tag_id in sorted(tag2items.keys()):
    rel = tag2rel[tag_id]
    items_list = sorted(tag2items[tag_id])
    if len(items_list) > MAX_ITEMS_PER_TAG:
        random.seed(tag_id)
        items_list = sorted(random.sample(items_list, MAX_ITEMS_PER_TAG))
    for i in range(len(items_list)):
        for j in range(i + 1, len(items_list)):
            a, b = items_list[i], items_list[j]
            for pair in [(a, b), (b, a)]:
                if pair not in pair2rel or REL_PRIORITY[rel] < REL_PRIORITY[pair2rel[pair]]:
                    pair2rel[pair] = rel

triplets = []
for (h, t), rel in pair2rel.items():
    triplets.append((h, rel, t))

rel_counts = defaultdict(int)
for _, rel, _ in triplets:
    rel_counts[rel] += 1
for rel_id in range(N_RELATIONS):
    print(f"[info] {rel_names[rel_id]}: {rel_counts.get(rel_id, 0)} triplets", flush=True)
print(f"[info] total triplets before balance: {len(triplets)}", flush=True)

# ── 9. 按关系类型采样平衡 ──
MAX_PER_REL = 50000
rel_triplets = defaultdict(list)
for t in triplets:
    rel_triplets[t[1]].append(t)

balanced_triplets = []
for rel_id in range(N_RELATIONS):
    rel_list = rel_triplets[rel_id]
    if len(rel_list) > MAX_PER_REL:
        random.seed(42)
        rel_list = random.sample(rel_list, MAX_PER_REL)
    balanced_triplets.extend(rel_list)
    print(f"[info] {rel_names[rel_id]}: {len(rel_list)} after balance (cap={MAX_PER_REL})", flush=True)

# ── 10. 写出 triplets.txt ──
balanced_triplets.sort(key=lambda x: (x[0], x[1], x[2]))
with open(out_triplets, 'w', encoding='utf-8') as f:
    for h, r, t in balanced_triplets:
        f.write(f"{h}\t{r}\t{t}\n")

print(f"[done] wrote {len(balanced_triplets)} triplets to {out_triplets}", flush=True)
print(f"[done] wrote item2entity to {out_item2entity}", flush=True)

# ── 11. 统计摘要 ──
head_items = set(t[0] for t in balanced_triplets)
tail_items = set(t[2] for t in balanced_triplets)
rel_set = set(t[1] for t in balanced_triplets)
print(f"[stats] head items: {len(head_items)}, tail items: {len(tail_items)}", flush=True)
print(f"[stats] relations used: {sorted(rel_set)} ({len(rel_set)} types)", flush=True)
if head_items:
    print(f"[stats] head range: {min(head_items)}~{max(head_items)}", flush=True)
if tail_items:
    print(f"[stats] tail range: {min(tail_items)}~{max(tail_items)}", flush=True)
print(f"[stats] all head/tail within [0, {num_items - 1}]: "
      f"{all(0 <= v < num_items for v in head_items | tail_items)}", flush=True)
