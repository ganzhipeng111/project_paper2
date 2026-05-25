import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import utils



class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)  

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma

class CoLaKG(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset, 
                 adj_matrix=None, 
                 adj_relations=None,
                 n_relations=None,
                 semantic_emb=None, 
                 user_semantic_emb=None,):
        super(CoLaKG, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.adj_matrix = adj_matrix.to(world.device)
        # 处理关系矩阵
        if adj_relations is None:
            # 如果未提供，创建一个全0的占位矩阵（假设只有一种默认关系）
            self.adj_relations = torch.zeros_like(self.adj_matrix).to(world.device)
            self.n_relations = 1
        else:
            self.adj_relations = adj_relations.to(world.device)
            self.n_relations = n_relations if n_relations is not None else (self.adj_relations.max().item() + 1)
            
        self.semantic_emb = semantic_emb.to(world.device)
        self.user_semantic_emb = user_semantic_emb.to(world.device)
        self.semantic_hid = 32
        self.dropout_i = self.config['dropout_i']
        self.dropout_u = self.config['dropout_u']
        self.dropout_neighbor = self.config['dropout_n']
        
        self.latent_dim = self.config['latent_dim_rec']
        self.kg_mask_prob = self.config.get('kg_mask_prob', 0.1)
        self.consistency_update_interval = self.config.get('consistency_update_interval')
        self.stability_scores = None
        self.current_epoch = 0
        self.kg_warmup_epochs = int(self.config.get('kg_warmup_epochs', 30))

        self.relation_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.dynamic_intent_lambda = self.config.get('dynamic_intent_lambda')
        self.kg_stability_lambda = self.config.get('kg_stability_lambda')
        self.stability_alpha = self.config.get('stability_alpha')
        self.stability_start_epoch = self.config.get('stability_start_epoch')
        
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        print("self.num_items", self.num_items)
        self._item_arange = torch.arange(self.num_items, device=world.device)
        # 删除重复：不在这里重新赋值 latent_dim
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')
        
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.semantic_map = nn.Linear(1024, self.latent_dim)
        self.user_semantic_map = nn.Linear(1024, self.latent_dim)
        print(f"lgn is already to go(drop_edge:{self.config['use_drop_edge']})")

        self.kgin_agg = KGINAggregatorLite(
            n_relations=self.n_relations,
            latent_dim=self.latent_dim,
            activation='elu', 
            dropout=self.dropout_neighbor
        )

        # 细粒度门控 MLP + Sigmoid + 温度系数（逐维门控）
        self.gate_temperature = float(self.config.get('gate_temperature', 1.2))
        self.item_fusion_gate = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.user_fusion_gate = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        with torch.no_grad():
            # 让初始门控在 0.5 附近，更稳
            self.item_fusion_gate[-1].bias.zero_()
            self.user_fusion_gate[-1].bias.zero_()
        
        # -----------------------------------------------------------
        # 新增：用户意图建模与独立性约束组件 (KGIN Style)
        # -----------------------------------------------------------
        self.n_factors = int(self.config.get('n_factors', 4))  # 意图数量
        self.ind = self.config.get('ind', 'distance')          # 独立性度量方式
        self.sim_decay = float(self.config.get('sim_regularity', 1e-4)) # 独立性约束权重
        
        # 潜在意图矩阵 [n_factors, latent_dim]
        self.latent_emb = nn.Parameter(torch.empty(self.n_factors, self.latent_dim))
        nn.init.xavier_uniform_(self.latent_emb)
        
        # 意图解耦注意力参数 [n_factors, latent_dim] (此处简化：直接映射到 channel 维度)
        # KGIN 原版是 [n_factors, n_relations]，这里我们直接用 [n_factors, latent_dim] 做自适应加权
        self.disen_weight_att = nn.Parameter(torch.empty(self.n_factors, self.latent_dim))
        nn.init.xavier_uniform_(self.disen_weight_att)
        # 删除旧注意力参数（未使用）
        # self.W = nn.Parameter(torch.empty(size=(1024, 32)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = nn.Parameter(torch.empty(size=(2*32, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.alpha=0.2
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def generate_masked_kg_views(self, adj_matrix):
        """
        生成两个随机掩码的知识图谱索引子视图（保持long索引类型）
        """
        # 创建掩码矩阵
        N, L = adj_matrix.size(0), adj_matrix.size(1)
        keep1 = (torch.rand(N, L, device=adj_matrix.device) > self.kg_mask_prob)
        keep2 = (torch.rand(N, L, device=adj_matrix.device) > self.kg_mask_prob)
        self_idx = torch.arange(N, device=adj_matrix.device).unsqueeze(1).expand(N, L)
        view1 = torch.where(keep1, adj_matrix, self_idx).long()
        view2 = torch.where(keep2, adj_matrix, self_idx).long()
        
        return view1, view2

    def compute_consistency_scores(self, item_embeddings_view1, item_embeddings_view2):
        """
        计算同一物品在两个子视图下的表示结构一致性分数
        """
        # 计算余弦相似度作为一致性分数
        cosine_sim = F.cosine_similarity(item_embeddings_view1, item_embeddings_view2, dim=-1)
        # 将相似度映射到0-1范围
        consistency_scores = (cosine_sim + 1) / 2
        return consistency_scores

    def update_stability_scores(self):
        if (
            self.consistency_update_interval is not None and
            self.stability_start_epoch is not None and
            self.consistency_update_interval > 0 and
            self.current_epoch >= self.stability_start_epoch and
            self.current_epoch % self.consistency_update_interval == 0
        ):
            with torch.no_grad():
                item_norms = self.embedding_item.weight.norm(p=2, dim=-1)
                min_norm = item_norms.min()
                max_norm = item_norms.max()
                if max_norm > min_norm:
                    consistency_scores = (item_norms - min_norm) / (max_norm - min_norm)
                else:
                    consistency_scores = torch.ones_like(item_norms)
                
                if self.stability_scores is None:
                    self.stability_scores = consistency_scores
                else:
                    alpha = self.stability_alpha
                    self.stability_scores = alpha * consistency_scores + (1 - alpha) * self.stability_scores

    def get_stability_weights(self, item_indices):
        """
        获取物品的稳定性权重
        """
        if self.stability_scores is None:
            return torch.ones_like(item_indices, dtype=torch.float32)
        
        stability_weights = self.stability_scores[item_indices]
        # 确保权重在合理范围内
        stability_weights = torch.clamp(stability_weights, 0.1, 1.0)
        return stability_weights

    # [MODIFIED] 动态关系权重计算 (Global Relation Level)
    def compute_dynamic_relation_weights(self):
        """
        基于关系嵌入计算每种关系的动态权重
        Returns: [n_relations]
        """
        # 获取所有关系的嵌入 [n_relations, latent_dim]
        # 注意：需要从 aggregator 中获取关系嵌入层
        relation_embs = self.kgin_agg.relation_emb.weight
        
        # 计算权重 [n_relations, 1] -> [n_relations]
        weights = self.relation_mlp(relation_embs).squeeze(-1)
        
        return weights

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def _cul_cor(self):
        """
        计算意图独立性损失 (Independence Constraint)
        """
        def CosineSimilarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2

        def DistanceCorrelation(tensor_1, tensor_2):
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            a_ = torch.matmul(tensor_1, tensor_1.t()) * 2
            b_ = torch.matmul(tensor_2, tensor_2.t()) * 2
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8)
            b = torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, torch.zeros(1).to(tensor_1.device)) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, torch.zeros(1).to(tensor_1.device)) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, torch.zeros(1).to(tensor_1.device)) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        cor = 0
        # 计算 latent_emb 或 disen_weight_att 之间的相关性
        # 这里选择约束 latent_emb，让意图中心尽可能正交
        target_matrix = self.latent_emb
        
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                if self.ind == 'distance':
                    cor += DistanceCorrelation(target_matrix[i], target_matrix[j])
                else:
                    cor += CosineSimilarity(target_matrix[i], target_matrix[j])
        return cor

    def _compute_warmup_factor(self):
        if self.kg_warmup_epochs <= 0:
            return 1.0
        progress = min(1.0, self.current_epoch / self.kg_warmup_epochs)
        return progress * progress

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        warmup = self._compute_warmup_factor()

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['use_drop_edge']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        if warmup > 0.01:
            items_semantic_emb = F.dropout(self.semantic_emb, self.dropout_i, training=self.training)
            items_semantic_emb = self.semantic_map(items_semantic_emb)
            items_semantic_emb = F.elu(items_semantic_emb)
            items_semantic_emb = F.dropout(items_semantic_emb, self.dropout_i, training=self.training)
            items_semantic_emb = F.normalize(items_semantic_emb, p=2, dim=-1)

            user_semantic_emb = F.dropout(self.user_semantic_emb, self.dropout_u, training=self.training)
            user_semantic_emb = self.user_semantic_map(user_semantic_emb)
            user_semantic_emb = F.elu(user_semantic_emb)
            user_semantic_emb = F.dropout(user_semantic_emb, self.dropout_u, training=self.training)
            user_semantic_emb = F.normalize(user_semantic_emb, p=2, dim=-1)

            item_gate_input = torch.cat([items, items_semantic_emb], dim=-1)
            item_gate = torch.sigmoid(self.item_fusion_gate(item_gate_input) / self.gate_temperature)
            items = item_gate * items_semantic_emb + (1.0 - item_gate) * items

            user_gate_input = torch.cat([users, user_semantic_emb], dim=-1)
            user_gate = torch.sigmoid(self.user_fusion_gate(user_gate_input) / self.gate_temperature)
            users = user_gate * user_semantic_emb + (1.0 - user_gate) * users

            kg_n_hops = int(self.config.get('kg_n_hops'))
            kg_hop_decay = float(self.config.get('kg_hop_decay'))
            use_residual = bool(self.config.get('kg_hop_residual'))

            dynamic_rel_weights = None
            if hasattr(self, 'dynamic_intent_lambda') and self.dynamic_intent_lambda > 0:
                dynamic_rel_weights = self.compute_dynamic_relation_weights()

            h_prev = items
            h_list = []
            for t in range(kg_n_hops):
                h_next = self.kgin_agg.aggregate(
                    h_prev,
                    self.adj_matrix,
                    self.adj_relations,
                    self.training,
                    dynamic_weights=dynamic_rel_weights
                )
                h_prev = (h_prev + h_next) / 2 if use_residual else h_next
                h_list.append(h_prev)

            weights = [kg_hop_decay ** t for t in range(kg_n_hops)]
            denom = sum(weights)
            h_prime = sum(w * h for w, h in zip(weights, h_list)) / denom

            if self.stability_scores is not None:
                stability_weights = self.get_stability_weights(self._item_arange)
                factor = 1.0 + self.kg_stability_lambda * (stability_weights.unsqueeze(-1) - 1.0)
                h_prime = h_prime * factor
            items = items + h_prime * warmup

            score_ = torch.mm(users, self.latent_emb.t())
            score = nn.Softmax(dim=1)(score_)
            intent_context = torch.sum(score.unsqueeze(-1) * self.disen_weight_att.unsqueeze(0), dim=1)

            if self.stability_scores is not None:
                if not hasattr(self, '_stability_vector') or self._stability_vector is None:
                    n_nodes = self.num_users + self.num_items
                    self._stability_vector = torch.zeros(n_nodes, 1, device=world.device)
                self._stability_vector.zero_()
                self._stability_vector[self.num_users:] = self.stability_scores.unsqueeze(1)

                if self.A_split:
                    user_consistency = torch.zeros(self.num_users, 1, device=world.device)
                    for g in self.Graph:
                        prop = torch.sparse.mm(g, self._stability_vector)
                        user_consistency += prop[:self.num_users]
                else:
                    prop = torch.sparse.mm(self.Graph, self._stability_vector)
                    user_consistency = prop[:self.num_users]

                intent_context = intent_context * user_consistency

            users = users + intent_context * warmup

        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        scores = torch.matmul(users_emb, items_emb.t())
        # [REMOVED] 旧的动态意图 gating
        rating = self.f(scores)
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) + 
                            posEmb0.norm(2).pow(2) + 
                            negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        bpr = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        warmup = self._compute_warmup_factor()
        if warmup > 0.01:
            cor_loss = self.sim_decay * self._cul_cor() * warmup
            loss = bpr + cor_loss
        else:
            loss = bpr
        
        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        # [REMOVED] 旧的动态权重后处理逻辑
        return gamma

class KGINAggregatorLite(nn.Module):
    def __init__(self, n_relations: int, latent_dim: int, activation: str = 'elu', dropout: float = 0.1):
        super().__init__()
        self.n_relations = n_relations
        self.latent_dim = latent_dim
        self.activation = activation
        self.dropout = dropout
        
        self.relation_emb = nn.Embedding(n_relations, latent_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.attention_hid = 16
        self.W_q = nn.Linear(latent_dim, self.attention_hid, bias=False)
        self.W_k = nn.Linear(latent_dim, self.attention_hid, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def aggregate(self, entity_emb: torch.Tensor, adj_indices: torch.Tensor, adj_relations: torch.Tensor, training: bool, dynamic_weights: torch.Tensor = None) -> torch.Tensor:
        N, L = adj_indices.shape
        
        neighbors = entity_emb[adj_indices]
        relations = self.relation_emb(adj_relations)
        
        if dynamic_weights is not None:
            relations = relations * dynamic_weights[adj_relations].unsqueeze(-1)
        
        neighbors_agg = neighbors * relations
        
        center_q = self.W_q(entity_emb).unsqueeze(1)
        neighbor_k = self.W_k(neighbors)
        attention = (center_q * neighbor_k).sum(dim=-1) / (self.attention_hid ** 0.5)
        attention = self.leakyrelu(attention)
        attention_score = F.softmax(attention, dim=1)
        attention_score = F.dropout(attention_score, self.dropout, training=training)
        
        out = torch.bmm(attention_score.unsqueeze(1), neighbors_agg).squeeze(1)
        
        if self.activation == 'elu':
            out = F.elu(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        
        out = F.dropout(out, self.dropout, training=training)
        
        return out