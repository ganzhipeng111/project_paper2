---
name: "lastfm-kg-rebuild"
description: "Rebuild LastFM KG triplets with unified ID space. Invoke when modifying build_lastfm_duoguanxitu.py or fixing KG data quality for CoLaKG model."
---

# LastFM KG 数据重构

## 自更新协议

修改下表文件后，必须同步更新本文件对应章节。

| 触发文件 | 更新章节 |
|----------|---------|
| `data_preprocess/build_lastfm_duoguanxitu.py` | 代码状态、格式规范、变更日志 |
| `data_preprocess/build_lastfm_item_neighbors_from_kg.py` | 代码状态、变更日志 |
| `rec_code/main.py` (L72-121) | 代码状态、变更日志 |
| `data/lastfm/triplets.txt` | 数据状态、变更日志 |
| `data/lastfm/item2entity.txt` | 数据状态、变更日志 |
| `data/lastfm/lastfm_neighbors_kg.pt` | 数据状态、变更日志 |

规则: ①追加变更日志(限最近10条) ②更新状态标记 ③更新数据统计 ④原则/规范变化时同步更新

## 当前状态

### 代码

| 文件 | 状态 |
|------|------|
| `build_lastfm_duoguanxitu.py` | ✅已重构 |
| `build_lastfm_item_neighbors_from_kg.py` | ✅已重构 |
| `main.py` (adj_relations) | ✅无需修改 |

### 数据

| 文件 | 状态 | 统计 |
|------|------|------|
| `triplets.txt` | ✅ | 194,988条 / 5种关系 / head&tail=item_id 0~2812 |
| `item2entity.txt` | ✅ | 2,813条 / entity_id=item_id 1:1映射 |
| `lastfm_neighbors_kg.pt` | ✅ | (2813,10) / non-self-loop 97.0% |

## 变更日志

| 日期 | 操作 | 备注 |
|------|------|------|
| 2026-05-25 | 重构build_lastfm_duoguanxitu.py | 修正ID映射、统一ID空间、tag语义分类5类、item-item关系 |
| 2026-05-25 | 重构build_lastfm_item_neighbors_from_kg.py | 适配新triplets格式、直接item ID共现 |
| 2026-05-25 | 确认main.py无需修改 | Loader推断m_items=2813、triplets head/tail均为item ID |
| 2026-05-25 | 创建SKILL | 识别4个核心问题 |

## 设计原则

1. **统一ID空间**: triplets.txt head和tail均为rec item ID(0~2812)，与main.py relation_dict[(head,tail)]查询对齐
2. **细粒度关系**: tag按语义分5类(genre/mood/era/region/other)，同一(item,tail)对保留最高优先级关系
3. **去重平衡**: 每tag最多50个item、每关系类型上限50000条
4. **main.py对齐**: triplets.txt直接被main.py消费，无需entity_id转换

## 格式规范

- `item_map.txt`: `artistID itemID`（左artist右item，不修改）
- `triplets.txt`: `head_item_id \t relation_id \t tail_item_id`（head和tail均为rec item ID）
- `item2entity.txt`: `item_id \t entity_id`（item实体: entity_id=item_id，1:1映射）
- `lastfm_neighbors_kg.pt`: shape=(num_items, neighbor_k)，值为item ID

## 关键数据参数

- num_items = 2813 (从item_map.txt推断，与train.txt一致)
- n_users = 1859
- 关系类型: 0=genre, 1=mood, 2=era, 3=region, 4=other
- item_map是artist→item 1:1映射，无法建立byArtist的item-item对
