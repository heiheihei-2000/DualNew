import torch
import torch.nn as nn
from utils.drug_source import variadic_topk
from torch_scatter import scatter
import numpy as np


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, use_lama_rel, K, sample_flag, act=lambda x: x, use_adaptive_k: bool = False, use_relation_gating: bool = False):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_lama_rel = use_lama_rel
        self.K = K
        self.sample_flag = sample_flag
        # 可选开关（默认关闭以保持速度）
        self.use_adaptive_k = use_adaptive_k
        self.use_relation_gating = use_relation_gating

        self.Ws_attn = nn.Linear(in_dim, attn_dim)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wq_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        
        # 关系感知门控（可选）
        if self.use_relation_gating:
            self.relation_gates = nn.Parameter(torch.randn(n_rel * 2 + 1, in_dim))
            self.relation_importance = nn.Linear(in_dim * 2, 1)
        else:
            self.relation_gates = None
            self.relation_importance = None

    def forward(self, q_sub, q_rel, q_emb, rela_embed, hidden, edges, nodes, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx] # q_rel 代表问题的id
        l1 = edges.shape[0]
        n1 = nodes.size(0)
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        # print(edges.shape[0])
        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel, :]
        else:
            hr = rela_embed(rel)

        self.n_rel = (rela_embed.shape[0] - 1) // 2

        r_idx = edges[:, 0]
        h_qr = q_emb[edges[:, 0], :]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(
            nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr) + self.Wqr_attn(hr * h_qr))))

        sample_flag = self.sample_flag
        # ========= alpha + [0:2]=============
        if sample_flag == 1:
            if self.use_adaptive_k:
                # 慢：自适应K（逐源循环）
                _, ind1 = torch.unique(edges[:, 0:2], dim=0, sorted=True, return_inverse=True)
                _, ind2 = torch.sort(ind1)
                edges = edges[ind2]  # sort edges
                alpha = alpha[ind2]
                _, counts = torch.unique(edges[:, 0:2], dim=0, return_counts=True)

                idd_idx = edges[:, 2] == (self.n_rel * 2)
                idd_edges = edges[idd_idx]

                probs = alpha.squeeze()

                mask_indices = []
                start_idx = 0
                for count in counts:
                    end_idx = start_idx + count
                    node_probs = probs[start_idx:end_idx]
                    if count > 1:
                        node_probs_norm = torch.softmax(node_probs, dim=0)
                        entropy = -(node_probs_norm * torch.log(node_probs_norm + 1e-10)).sum()
                        entropy_factor = 0.5 + torch.sigmoid(entropy).item()
                        adaptive_k = min(int(self.K * entropy_factor), count.item())
                    else:
                        adaptive_k = 1
                    if adaptive_k > 0:
                        _, topk_idx = torch.topk(node_probs, min(adaptive_k, count.item()))
                        mask_indices.extend((start_idx + topk_idx).tolist())
                    start_idx = end_idx

                mask = torch.tensor(mask_indices, dtype=torch.long, device=edges.device)
                edges = edges[mask]
                edges = torch.cat((edges, idd_edges), 0)
                edges = torch.unique(edges[:, :], dim=0)

                nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
                edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], 1)

                head_index = edges[:, 4]
                idd_mask = edges[:, 2] == (self.n_rel * 2)
                _, old_idx = head_index[idd_mask].sort()
                old_nodes_new_idx = tail_index[idd_mask][old_idx]
            else:
                # 快：原始 variadic_topk
                max_ent_per_ent = self.K
                _, ind1 = torch.unique(edges[:, 0:2], dim=0, sorted=True, return_inverse=True)
                _, ind2 = torch.sort(ind1)
                edges = edges[ind2]  # sort edges
                alpha = alpha[ind2]
                _, counts = torch.unique(edges[:, 0:2], dim=0, return_counts=True)
                idd_idx = edges[:, 2] == (self.n_rel * 2)
                idd_edges = edges[idd_idx]

                probs = alpha.squeeze()
                topk_value, topk_index = variadic_topk(probs, counts, k=max_ent_per_ent)

                cnt_sum = torch.cumsum(counts, dim=0)
                cnt_sum[1:] = cnt_sum[:-1] + 0
                cnt_sum[0] = 0
                topk_index = topk_index + cnt_sum.unsqueeze(1)

                mask = topk_index.view(-1, 1).squeeze()
                mask = torch.unique(mask)

                edges = edges[mask]
                edges = torch.cat((edges, idd_edges), 0)
                edges = torch.unique(edges[:, :], dim=0)

                nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
                edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], 1)

                head_index = edges[:, 4]
                idd_mask = edges[:, 2] == (self.n_rel * 2)
                _, old_idx = head_index[idd_mask].sort()
                old_nodes_new_idx = tail_index[idd_mask][old_idx]

        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]
        # print(edges.shape[0])
        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel, :]
        else:
            hr = rela_embed(rel)

        r_idx = edges[:, 0]
        h_qr = q_emb[edges[:, 0], :]

        # 消息传递（可选关系门控）
        if self.use_relation_gating and (self.relation_gates is not None):
            gate = torch.sigmoid(self.relation_gates[rel])  # [n_edges, hidden_dim]
            importance = torch.sigmoid(self.relation_importance(torch.cat([hr, h_qr], dim=-1)))  # [n_edges, 1]
            message = hs * gate * hr * importance
            alpha = torch.sigmoid(self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr) + self.Wqr_attn(hr * h_qr))))
            message = alpha * message
        else:
            message = hs * hr
            alpha = torch.sigmoid(self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr) + self.Wqr_attn(hr * h_qr))))
            message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.size(0), reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))
        # print(nodes.shape, hidden_new.shape)
        l2 = edges.shape[0]
        n2 = nodes.size(0)
        num_node = np.array([n1 * 1.0 / len(q_sub), n2 * 1.0 / len(q_sub)])
        num_edge = np.array([l1 * 1.0 / len(q_sub), l2 * 1.0 / len(q_sub)])

        return num_node, num_edge, hidden_new, alpha, nodes, edges, old_nodes_new_idx


class Explore(torch.nn.Module):
    def __init__(self, params, loader, device=None):
        super(Explore, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        # Store device for later use
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.K = params.K
        self.sample_flag = params.sample

        self.question_emb = self.load_qemb().detach().to(self.device)
        # self.W_q = nn.Linear(5120,self.hidden_dim)
        self.dim_reduct = nn.Sequential(
            nn.Linear(5120, 2096),
            nn.ReLU(),
            nn.Linear(2096, self.hidden_dim)
        ).to(self.device)  # Move to the correct device
        self.use_lama_rel = 1
        if self.use_lama_rel == 1:
            self.rela_embed = self.load_rel_emb().detach().to(self.device)
        else:
            self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        self.gnn_layers = []
        for i in range(3):
            use_adaptive_k_first = (getattr(params, 'use_adaptive_k', True) and i == 0)
            self.gnn_layers.append(
                GNNLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.attn_dim,
                    self.n_rel,
                    self.use_lama_rel,
                    self.K,
                    self.sample_flag,
                    act=act,
                    use_adaptive_k=use_adaptive_k_first,
                    use_relation_gating=getattr(params, 'use_relation_gating', False),
                )
            )
        # self.gnn_layers = nn.ModuleList(self.gnn_layers)
        # # Move all GNN layers to the correct device
        # for layer in self.gnn_layers:
        #     layer.to(self.device)
        self.gnn_layers = nn.ModuleList(self.gnn_layers).to(self.device)
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False).to(self.device)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim).to(self.device)
        self.Wq_final = nn.Linear(self.hidden_dim * 2, 1, bias=False).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 1)
        ).to(self.device)
        # 轻量路径感知评分：对最后一层关系向量按注意力聚合，作为加分项
        self.path_proj = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.path_coef = 0.2  # 加权系数，可在需要时调节
        
        self.Wr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True).to(self.device)  # r^-1 = Wr+b
        self.loop = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, subs, qids, mode='train', question_texts=None):
        n_qs = len(qids)
        # q_sub = subs
        # q_id = torch.LongTensor(qids)  # .cuda()
        q_sub = subs

        # qids -> 与 question_emb 同设备，避免索引设备不一致
        q_id = torch.as_tensor(qids, dtype=torch.long, device=self.question_emb.device)
        ques_emb = self.question_emb[q_id]
        q_emb = self.dim_reduct(ques_emb)

        if self.use_lama_rel == 1:
            # rela_embed 已在 __init__/change_loader 中放到 device
            rel_emb = self.dim_reduct(self.rela_embed)

            rel_emb = rel_emb[0:self.n_rel, :]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop.to(self.device)], dim=0)

        else:
            rel_emb = self.rela_embed

        n_node = sum(len(sublist) for sublist in subs)
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),
            np.concatenate(subs)
        ]).reshape(2, -1)
        nodes = np.array(nodes, dtype=np.int64)
        nodes = torch.as_tensor(nodes, dtype=torch.long, device=self.device).T

        # h0 = torch.zeros((1, n_node, self.hidden_dim)).to(self.device)
        # # nodes = torch.cat([torch.arange(n).unsqueeze(1).to(self.device), q_sub.unsqueeze(1)], 1)
        # hidden = torch.zeros(n_node, self.hidden_dim).to(self.device)
        h0 = torch.zeros((1, n_node, self.hidden_dim), device=self.device)
        hidden = torch.zeros(n_node, self.hidden_dim, device=self.device)
        # hq init hs
        hidden = q_emb[nodes[:, 0], :]

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        scores_all = []
        
        # 方案5：保存每层的路径信息用于路径感知评分
        layer_hidden_states = []  # 保存每层的隐藏状态（按各层自身节点顺序）
        layer_edges = []          # 保存每层的边信息
        layer_mappings = []       # 保存上一层到当前层的节点映射 old_nodes_new_idx（用于对齐）
        
        for i in range(self.n_layer):
            # nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids, device=self.device)
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(
                            nodes.data.cpu().numpy(), qids, device=self.device
                                                                        )
            num_node, num_edge, hidden, alpha, nodes, edges, old_nodes_new_idx = self.gnn_layers[i](q_sub, q_id, q_emb,
                                                                                                    rel_emb, hidden,
                                                                                                    edges, nodes,
                                                                                                    old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.device).index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            
            # 保存每层的信息
            layer_hidden_states.append(hidden)
            layer_edges.append(edges)
            layer_mappings.append(old_nodes_new_idx)

            num_nodes[i, :] += num_node
            num_edges[i, :] += num_edge

        h_qs = q_emb[nodes[:, 0], :]
        
        # 方案5：轻量路径感知评分
        base_scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        # 使用最后一层的边和注意力，按目标节点归一化聚合关系向量作为路径摘要
        # 注：last_edges 与 last_alpha 在上面的循环末尾可被记录，这里用当前层的 edges/alpha 近似
        rel_idx_last = edges[:, 2]
        obj_last = edges[:, 5]
        rel_vec_last = rel_emb[rel_idx_last, :]
        alpha_w = alpha.squeeze()
        sum_per_obj = scatter(alpha_w, index=obj_last, dim=0, dim_size=nodes.size(0), reduce='sum')
        denom = sum_per_obj[obj_last] + 1e-9
        alpha_norm = alpha_w / denom
        path_rel = scatter(alpha_norm.unsqueeze(1) * rel_vec_last, index=obj_last, dim=0, dim_size=nodes.size(0), reduce='sum')
        path_bias = self.path_proj(path_rel).squeeze(-1)
        scores = base_scores + self.path_coef * path_bias
        
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).to(self.device)
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        
        # 实现h_g的均值池化：h_g = MEAN{h(v)}
        # hidden: [num_nodes, hidden_dim] -> h_g_pooled: [batch_size, hidden_dim]
        h_g_pooled = torch.zeros(n_qs, self.hidden_dim).to(self.device)
        for i in range(n_qs):
            # 找到属于第i个问题的节点
            question_nodes_mask = nodes[:, 0] == i
            if question_nodes_mask.sum() > 0:
                # 对属于该问题的节点嵌入进行均值池化
                h_g_pooled[i] = torch.mean(hidden[question_nodes_mask], dim=0)
        

        
        # 生成子图结果用于返回（不再生成多选提示）
        processed_subgraph = ""
        # 兼容两套调用：
        # - GraphLLM 使用 llm_train/llm_inference，返回 (h_g_pooled, processed_subgraph, scores_all)
        # - 旧版 BaseModel/train.py 使用 train/valid/test，返回 (num_nodes, num_edges, scores_all, _, _, _)
        if mode in ('llm_train', 'llm_inference'):
            return h_g_pooled, processed_subgraph, scores_all
        else:
            # 兼容旧评估调用，返回6元组（后3项占位）
            return num_nodes, num_edges, scores_all




    def load_qemb(self):

        datapath = self.loader.task_dir
        if 'MetaQA/1-hop' in datapath:
            q_train = np.load('../embedding/Meta-1m-train.npy')
            q_valid = np.load('../embedding/Meta-1m-valid.npy')
            q_test = np.load('../embedding/Meta-1m-test.npy')
        elif 'MetaQA/2-hop' in datapath:
            q_train = np.load('../embedding/Meta-2m-train.npy')
            q_valid = np.load('../embedding/Meta-2m-valid.npy')
            q_test = np.load('../embedding/Meta-2m-test.npy')
        elif 'MetaQA/3-hop' in datapath:
            q_train = np.load('../embedding/Meta-3m-train.npy')
            q_valid = np.load('../embedding/Meta-3m-valid.npy')
            q_test = np.load('../embedding/Meta-3m-test.npy')
        elif 'webqsp' in datapath:
            q_train = np.load('../embedding/13b-webqsp/webqsp-train.npy')
            q_valid = np.load('../embedding/13b-webqsp/webqsp-valid.npy')
            q_test = np.load('../embedding/13b-webqsp/webqsp-test.npy')
        elif 'CWQ' in datapath:
            q_train = np.load('../embedding/13b-CWQ/CWQ-train.npy')
            q_valid = np.load('../embedding/13b-CWQ/CWQ-valid.npy')
            q_test = np.load('../embedding/13b-CWQ/CWQ-test.npy')

        q_emb = np.concatenate((q_train, q_valid, q_test))

        return torch.tensor(q_emb, dtype=torch.float32)

    def load_rel_emb(self):

        datapath = self.loader.task_dir
        if 'MetaQA' in datapath:
            rel_emb = np.load('../embedding/Meta-rel.npy')
        elif 'webqsp' in datapath:
            rel_emb = np.load('../embedding/13b-webqsp/webqsp-rel.npy')
        elif 'CWQ' in datapath:
            rel_emb = np.load('../embedding/13b-CWQ/CWQ-rel.npy')

        print('rel_emb shape: ', rel_emb.shape)

        return torch.tensor(rel_emb, dtype=torch.float32)

    def change_loader(self, loader):

        self.loader = loader
        self.question_emb = self.load_qemb().detach().to(self.device)
        self.rela_embed = self.load_rel_emb().detach().to(self.device)
        self.n_rel = self.loader.n_rel
        print('change loader:', self.loader.task_dir)

    def visual_path(self, subs, qids, objs, filepath, mode='test'):
        # 计算问题的数量
        n_qs = len(qids)
        # 保存主题实体列表
        q_sub = subs
        # 将问题ID转换为长整型张量
        q_id = torch.LongTensor(qids)

        # 根据问题ID获取对应的问题嵌入向量
        ques_emb = self.question_emb[q_id, :]
        # 将问题嵌入移至GPU
        ques_emb = ques_emb.cuda()
        # 将问题ID移至GPU
        q_id = q_id.cuda()
        # 对问题嵌入进行维度降维
        q_emb = self.dim_reduct(ques_emb)
        # 将问题嵌入移回CPU（释放GPU内存）
        ques_emb.cpu()

        # 如果使用Llama关系嵌入
        if self.use_lama_rel == 1:
            # 将关系嵌入移至GPU
            self.rela_embed = self.rela_embed.cuda()
            # 对关系嵌入进行维度降维
            rel_emb = self.dim_reduct(self.rela_embed)
            # 将关系嵌入移回CPU
            self.rela_embed.cpu()

            # 截取前n_rel个关系嵌入（基础关系）
            rel_emb = rel_emb[0:self.n_rel, :]
            # 通过线性变换生成反向关系嵌入
            rev_rel_emb = self.Wr(rel_emb)
            # 拼接正向关系、反向关系和自环关系
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop.to(self.device)], dim=0)
        else:
            # 不使用Llama关系嵌入，直接使用预定义的关系嵌入
            rel_emb = self.rela_embed

        # 计算所有主题实体的总数
        n_node = sum(len(sublist) for sublist in subs)
        # 创建节点对：[问题索引, 实体ID]
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),  # 重复问题索引
            np.concatenate(subs)  # 拼接所有实体ID
        ]).reshape(2, -1)
        # 转换为int64类型
        nodes = np.array(nodes, dtype=np.int64)
        # 转换为PyTorch张量并移至GPU，转置为[节点数, 2]格式
        nodes = torch.LongTensor(nodes).T.cuda()

        # 初始化GRU的隐藏状态，维度为[1, 节点数, 隐藏维度]
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        # nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        # 初始化节点的隐藏表示
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        # 使用对应问题的嵌入初始化节点的隐藏表示
        hidden = q_emb[nodes[:, 0], :]

        # 记录每层的节点数（未使用）
        num_nodes = np.zeros((self.n_layer, 2))
        # 记录每层的边数（未使用）
        num_edges = np.zeros((self.n_layer, 2))

        # 存储所有层的节点信息
        all_nodes = []
        # 存储所有层的边信息
        all_edges = []
        # 存储所有层的边权重
        all_weights = []
        # 存储每层的最小权重
        min_weight = []

        # 遍历每个GNN层进行前向传播
        for i in range(self.n_layer):
            # 获取当前节点的邻居节点和边
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids, device=self.device)

            # 通过第i层GNN进行消息传递和聚合
            num_node, num_edge, hidden, weights, nodes, edges, old_nodes_new_idx = self.gnn_layers[i](q_sub, q_id,
                                                                                                      q_emb, rel_emb,
                                                                                                      hidden, edges,
                                                                                                      nodes,
                                                                                                      old_nodes_new_idx)
            # 更新GRU隐藏状态，保持旧节点的状态
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.device).index_copy_(1, old_nodes_new_idx, h0)

            # 对隐藏表示应用dropout正则化
            hidden = self.dropout(hidden)
            # 通过门控循环单元（GRU）更新隐藏状态
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)  #
            # 移除批次维度
            hidden = hidden.squeeze(0)
            # print(i,torch.max(weights),torch.min(weights))
            # 保存当前层的节点信息
            all_nodes.append(nodes.cpu().data.numpy())
            # 保存当前层的边信息
            all_edges.append(edges.cpu().data.numpy())
            # 保存当前层的权重信息
            all_weights.append(weights.cpu().data.numpy())
            # 记录最小权重值
            min_weight.append(torch.min(weights).item())

        # 获取最终节点对应的问题嵌入
        h_qs = q_emb[nodes[:, 0], :]
        # 通过MLP计算每个节点的得分
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        # 初始化所有实体的得分矩阵
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        # 将计算出的得分填入对应位置
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        # 转换为numpy数组并移至CPU
        scores_all = scores_all.squeeze().cpu().data.numpy()
        # 设置要返回的top-k答案数量
        n = 10
        # 按得分降序排序，获取前n个索引
        top_indices = np.argsort(scores_all)[::-1][:n]
        # 保存答案实体ID
        answer = top_indices

        # 对得分进行softmax归一化得到概率
        softscore = self.softmax(scores_all)
        # 获取前n个最高概率值
        top_values = np.partition(softscore, -2)[::-1][:n]
        # 保存概率值
        probs = top_values

        # 用于收集返回内容的字符串
        result_content = ""
        
        # 以追加模式打开文件
        f = open(filepath, 'a+')
        # 计算相对问题索引（减去训练集和验证集的问题数量）
        if mode == 'train':
            qs = qids
        elif mode == 'test':
            qs = qids-self.loader.n_train_qs-self.loader.n_valid_qs
        elif mode =='valid':
            qs = qids-self.loader.n_train_qs

        # 写入文件并收集到result_content
        # 写入问题索引作为行开头
        line_start = f'{qs[0]}\t'
        f.write(line_start)
        result_content += line_start

        # 遍历前n个答案候选
        for k in range(n):
            # 获取第k个答案实体ID
            tails = answer[k]
            # 格式化输出字符串（调试用）
            outstr = 'tail: %d,  p:%.2f' % (tails, probs[k])

            # 生成实体名称和概率的字符串
            entity_prob_str = '%s|%0.3f|' % (self.loader.id2entity[answer[k]], probs[k])
            # 写入文件
            f.write(entity_prob_str)
            # 同时收集到返回内容
            result_content += entity_prob_str
            
            # 用于存储推理路径的边
            print_edges = []
            # 从最后一层向前回溯，找到通向答案的路径
            for i in range(self.n_layer - 1, -1, -1):
                # print('layer:',i)
                # 获取第i层的边
                edges = all_edges[i]
                # print(edges.shape)
                # 获取第i层的权重
                weights = all_weights[i]
                # 找到尾节点等于当前答案节点的边
                mask1 = edges[:, 3] == tails
                # 如果没有找到，使用第一条边的尾节点
                if np.sum(mask1) == 0:
                    tails = edges[0, 3]
                    mask1 = edges[:, 3] == tails
                # 获取符合条件的边的权重
                weights1 = weights[mask1].reshape(-1, 1)
                # 获取符合条件的边
                edges1 = edges[mask1]
                # 找到权重最大的边的索引
                mask2 = np.argmax(weights1)

                # 选择权重最大的边
                new_edges = edges1[mask2].reshape(1, -1)
                # print(new_edges.shape)
                # 将权重保留两位小数
                new_weights = np.round_(weights1[mask2], 2).reshape(-1, 1)
                # print(new_weights.shape)
                # 拼接边信息：[头节点, 关系, 尾节点, 权重]
                new_edges = np.concatenate([new_edges[:, [1, 2, 3]], new_weights], 1)
                # new_edges: [h,r,t,alpha]
                # 更新尾节点为当前边的头节点（向前回溯）
                tails = new_edges[:, 0].astype('int')
                # 添加到路径列表
                print_edges.append(new_edges)

            # 从前向后输出路径（因为回溯是反向的）
            for i in range(self.n_layer - 1, -1, -1):
                # 获取第i层的边信息
                edge = print_edges[i][0].tolist()
                # 格式化输出字符串（调试用）
                outstr = '%d\t %d\t %d\t%.4f' % (edge[0], edge[1], edge[2], edge[3])

                # 处理正向关系（关系ID < n_rel）
                if edge[1] < self.loader.n_rel:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取关系名称
                    r = self.loader.id2relation[int(edge[1])]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 格式化为三元组字符串
                    edge_str = '(' + h + ', ' + r + ', ' + t + ');'
                    f.write(edge_str)
                    result_content += edge_str
                # 处理自环关系（关系ID == 2 * n_rel）
                elif edge[1] == 2 * self.n_rel:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取关系名称（自环）
                    r = self.loader.id2relation[int(edge[1])]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 格式化为三元组字符串
                    edge_str = '(' + h + ', ' + r + ', ' + t + ');'
                    f.write(edge_str)
                    result_content += edge_str
                # 处理反向关系（关系ID >= n_rel且不是自环）
                else:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取反向关系的原始关系名称
                    r = self.loader.id2relation[int(edge[1]) - self.loader.n_rel]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 反向关系需要交换头尾实体顺序
                    edge_str = '(' + t + ', ' + r + ', ' + h + ');'
                    f.write(edge_str)
                    result_content += edge_str
            # 添加制表符分隔
            tab_str = '\t'
            f.write(tab_str)
            result_content += tab_str
        
        # 添加换行符
        newline_str = '\n'
        f.write(newline_str)
        result_content += newline_str
        
        # 关闭文件
        f.close()

        # 处理all_nodes和all_edges为CSV格式
        processed_subgraph = self.process_all_nodes_edges_to_csv(all_nodes, all_edges)
        
        # 返回结果内容、所有节点、所有边和处理后的子图
        return result_content,all_nodes,all_edges,processed_subgraph

    def test_visual_path(self, subs, qids, objs, filepath, mode='test'):
        # 首先检查是否需要初始化已处理问题的集合
        if not hasattr(self, '_processed_questions'):
            self._processed_questions = set()
            self._last_filepath = None
        
        # 如果文件路径改变了（新文件），清空已处理问题集合
        if self._last_filepath != filepath:
            self._processed_questions = set()
            self._last_filepath = filepath
        
        # 对于训练集，使用问题ID本身（qids[0]是问题ID，不是索引）
        # 对于测试/验证集，计算相对问题索引
        if mode == 'train':
            # 训练集中qids[0]就是问题ID（来自train_data[id][1]）
            question_key = self.loader.n_train_qs
        elif mode == 'test':
            # 测试集使用相对索引
            relative_qid = qids[0] - self.loader.n_train_qs - self.loader.n_valid_qs
            question_key = f"test_{relative_qid}"
        else:  # valid
            # 验证集使用相对索引
            relative_qid = qids[0] - self.loader.n_train_qs
            question_key = f"valid_{relative_qid}"
        
        # 如果这个问题已经处理过了，直接返回
        if question_key in self._processed_questions:
            return
        
        # 标记这个问题已经处理
        self._processed_questions.add(question_key)
        
        # 计算问题的数量
        n_qs = len(qids)
        # 保存主题实体列表
        q_sub = subs
        # 将问题ID转换为长整型张量
        q_id = torch.LongTensor(qids)

        # 根据问题ID获取对应的问题嵌入向量
        ques_emb = self.question_emb[q_id, :]
        # 将问题嵌入移至GPU
        ques_emb = ques_emb.cuda()
        # 将问题ID移至GPU
        q_id = q_id.cuda()
        # 对问题嵌入进行维度降维
        q_emb = self.dim_reduct(ques_emb)
        # 将问题嵌入移回CPU（释放GPU内存）
        ques_emb.cpu()

        # 如果使用Llama关系嵌入
        if self.use_lama_rel == 1:
            # 将关系嵌入移至GPU
            self.rela_embed = self.rela_embed.cuda()
            # 对关系嵌入进行维度降维
            rel_emb = self.dim_reduct(self.rela_embed)
            # 将关系嵌入移回CPU
            self.rela_embed.cpu()

            # 截取前n_rel个关系嵌入（基础关系）
            rel_emb = rel_emb[0:self.n_rel, :]
            # 通过线性变换生成反向关系嵌入
            rev_rel_emb = self.Wr(rel_emb)
            # 拼接正向关系、反向关系和自环关系
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop.to(self.device)], dim=0)
        else:
            # 不使用Llama关系嵌入，直接使用预定义的关系嵌入
            rel_emb = self.rela_embed

        # 计算所有主题实体的总数
        n_node = sum(len(sublist) for sublist in subs)
        # 创建节点对：[问题索引, 实体ID]
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),  # 重复问题索引
            np.concatenate(subs)  # 拼接所有实体ID
        ]).reshape(2, -1)
        # 转换为int64类型
        nodes = np.array(nodes, dtype=np.int64)
        # 转换为PyTorch张量并移至GPU，转置为[节点数, 2]格式
        nodes = torch.LongTensor(nodes).T.cuda()

        # 初始化GRU的隐藏状态，维度为[1, 节点数, 隐藏维度]
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        # nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        # 初始化节点的隐藏表示
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        # 使用对应问题的嵌入初始化节点的隐藏表示
        hidden = q_emb[nodes[:, 0], :]

        # 记录每层的节点数（未使用）
        num_nodes = np.zeros((self.n_layer, 2))
        # 记录每层的边数（未使用）
        num_edges = np.zeros((self.n_layer, 2))

        # 存储所有层的节点信息
        all_nodes = []
        # 存储所有层的边信息
        all_edges = []
        # 存储所有层的边权重
        all_weights = []
        # 存储每层的最小权重
        min_weight = []

        # 遍历每个GNN层进行前向传播
        for i in range(self.n_layer):
            # 获取当前节点的邻居节点和边
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids, device=self.device)

            # 通过第i层GNN进行消息传递和聚合
            num_node, num_edge, hidden, weights, nodes, edges, old_nodes_new_idx = self.gnn_layers[i](q_sub, q_id,
                                                                                                      q_emb, rel_emb,
                                                                                                      hidden, edges,
                                                                                                      nodes,
                                                                                                      old_nodes_new_idx)
            # 更新GRU隐藏状态，保持旧节点的状态
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).to(self.device).index_copy_(1, old_nodes_new_idx, h0)

            # 对隐藏表示应用dropout正则化
            hidden = self.dropout(hidden)
            # 通过门控循环单元（GRU）更新隐藏状态
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)  #
            # 移除批次维度
            hidden = hidden.squeeze(0)
            # print(i,torch.max(weights),torch.min(weights))
            # 保存当前层的节点信息
            all_nodes.append(nodes.cpu().data.numpy())
            # 保存当前层的边信息
            all_edges.append(edges.cpu().data.numpy())
            # 保存当前层的权重信息
            all_weights.append(weights.cpu().data.numpy())
            # 记录最小权重值
            min_weight.append(torch.min(weights).item())

        # 获取最终节点对应的问题嵌入
        h_qs = q_emb[nodes[:, 0], :]
        # 通过MLP计算每个节点的得分
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        # 初始化所有实体的得分矩阵
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        # 将计算出的得分填入对应位置
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        # 转换为numpy数组并移至CPU
        scores_all = scores_all.squeeze().cpu().data.numpy()
        # 设置要返回的top-k答案数量
        n = 10
        # 按得分降序排序，获取前n个索引
        top_indices = np.argsort(scores_all)[::-1][:n]
        # 保存答案实体ID
        answer = top_indices

        # 对得分进行softmax归一化得到概率
        softscore = self.softmax(scores_all)
        # 获取前n个最高概率值
        top_values = np.partition(softscore, -2)[::-1][:n]
        # 保存概率值
        probs = top_values

        # 用于收集返回内容的字符串
        result_content = ""

        # 以追加模式打开文件
        f = open(filepath, 'a+')
        
        # 修正：计算相对问题索引
        if mode == 'train':
            qs = qids
            # 获取训练集的问题文本和答案
            question_text = self.loader.id2question.get(self.loader.train_q[qs[0]], f"Question ID: {qs[0]}")
            answer_entities = []
            if hasattr(self.loader, 'train_a') and qs[0] < len(self.loader.train_a):
                answer_ids = self.loader.train_a[qs[0]]
                if isinstance(answer_ids, (list, np.ndarray)):
                    for aid in answer_ids:
                        if aid in self.loader.id2entity:
                            answer_entities.append(self.loader.id2entity[aid])
                elif answer_ids in self.loader.id2entity:
                    answer_entities.append(self.loader.id2entity[answer_ids])
        elif mode == 'test':
            # 修正bug：正确计算测试集的相对索引
            qs = qids - self.loader.n_train_qs - self.loader.n_valid_qs
            # 获取测试集的问题文本和答案
            question_text = self.loader.id2question.get(self.loader.test_q[qs[0]], f"Question ID: {qs[0]}")
            answer_entities = []
            if hasattr(self.loader, 'test_a') and qs[0] < len(self.loader.test_a):
                answer_ids = self.loader.test_a[qs[0]]
                if isinstance(answer_ids, (list, np.ndarray)):
                    for aid in answer_ids:
                        if aid in self.loader.id2entity:
                            answer_entities.append(self.loader.id2entity[aid])
                elif answer_ids in self.loader.id2entity:
                    answer_entities.append(self.loader.id2entity[answer_ids])
        else:  # valid
            qs = qids - self.loader.n_train_qs
            # 获取验证集的问题文本和答案
            question_text = self.loader.id2question.get(self.loader.valid_q[qs[0]], f"Question ID: {qs[0]}")
            answer_entities = []
            if hasattr(self.loader, 'valid_a') and qs[0] < len(self.loader.valid_a):
                answer_ids = self.loader.valid_a[qs[0]]
                if isinstance(answer_ids, (list, np.ndarray)):
                    for aid in answer_ids:
                        if aid in self.loader.id2entity:
                            answer_entities.append(self.loader.id2entity[aid])
                elif answer_ids in self.loader.id2entity:
                    answer_entities.append(self.loader.id2entity[answer_ids])

        # 写入文件并收集到result_content
        # 先写入问题索引
        line_start = f'{qs[0]}\t'
        f.write(line_start)
        result_content += line_start
        
        # 写入问题文本
        question_info = f'[Q: {question_text}]\t'
        f.write(question_info)
        result_content += question_info
        
        # 写入正确答案
        answer_info = f'[A: {", ".join(answer_entities) if answer_entities else "Unknown"}]\t'
        f.write(answer_info)
        result_content += answer_info

        # 遍历前n个答案候选
        for k in range(n):
            # 获取第k个答案实体ID
            tails = answer[k]
            # 格式化输出字符串（调试用）
            outstr = 'tail: %d,  p:%.2f' % (tails, probs[k])

            # 生成实体名称和概率的字符串
            entity_prob_str = '%s|%0.3f|' % (self.loader.id2entity[answer[k]], probs[k])
            # 写入文件
            f.write(entity_prob_str)
            # 同时收集到返回内容
            result_content += entity_prob_str

            # 用于存储推理路径的边
            print_edges = []
            # 从最后一层向前回溯，找到通向答案的路径
            for i in range(self.n_layer - 1, -1, -1):
                # print('layer:',i)
                # 获取第i层的边
                edges = all_edges[i]
                # print(edges.shape)
                # 获取第i层的权重
                weights = all_weights[i]
                # 找到尾节点等于当前答案节点的边
                mask1 = edges[:, 3] == tails
                # 如果没有找到，使用第一条边的尾节点
                if np.sum(mask1) == 0:
                    tails = edges[0, 3]
                    mask1 = edges[:, 3] == tails
                # 获取符合条件的边的权重
                weights1 = weights[mask1].reshape(-1, 1)
                # 获取符合条件的边
                edges1 = edges[mask1]
                # 找到权重最大的边的索引
                mask2 = np.argmax(weights1)

                # 选择权重最大的边
                new_edges = edges1[mask2].reshape(1, -1)
                # print(new_edges.shape)
                # 将权重保留两位小数
                new_weights = np.round_(weights1[mask2], 2).reshape(-1, 1)
                # print(new_weights.shape)
                # 拼接边信息：[头节点, 关系, 尾节点, 权重]
                new_edges = np.concatenate([new_edges[:, [1, 2, 3]], new_weights], 1)
                # new_edges: [h,r,t,alpha]
                # 更新尾节点为当前边的头节点（向前回溯）
                tails = new_edges[:, 0].astype('int')
                # 添加到路径列表
                print_edges.append(new_edges)

            # 从前向后输出路径（因为回溯是反向的）
            for i in range(self.n_layer - 1, -1, -1):
                # 获取第i层的边信息
                edge = print_edges[i][0].tolist()
                # 格式化输出字符串（调试用）
                outstr = '%d\t %d\t %d\t%.4f' % (edge[0], edge[1], edge[2], edge[3])

                # 处理正向关系（关系ID < n_rel）
                if edge[1] < self.loader.n_rel:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取关系名称
                    r = self.loader.id2relation[int(edge[1])]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 格式化为三元组字符串
                    edge_str = '(' + h + ', ' + r + ', ' + t + ');'
                    f.write(edge_str)
                    result_content += edge_str
                # 处理自环关系（关系ID == 2 * n_rel）
                elif edge[1] == 2 * self.n_rel:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取关系名称（自环）
                    r = self.loader.id2relation[int(edge[1])]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 格式化为三元组字符串
                    edge_str = '(' + h + ', ' + r + ', ' + t + ');'
                    f.write(edge_str)
                    result_content += edge_str
                # 处理反向关系（关系ID >= n_rel且不是自环）
                else:
                    # 获取头实体名称
                    h = self.loader.id2entity[int(edge[0])]
                    # 获取反向关系的原始关系名称
                    r = self.loader.id2relation[int(edge[1]) - self.loader.n_rel]
                    # 获取尾实体名称
                    t = self.loader.id2entity[int(edge[2])]
                    # 反向关系需要交换头尾实体顺序
                    edge_str = '(' + t + ', ' + r + ', ' + h + ');'
                    f.write(edge_str)
                    result_content += edge_str
            # 添加制表符分隔
            tab_str = '\t'
            f.write(tab_str)
            result_content += tab_str

        # 添加换行符
        newline_str = '\n'
        f.write(newline_str)
        result_content += newline_str

        # 关闭文件
        f.close()

    def process_all_nodes_edges_to_csv(self, all_nodes, all_edges):
        """
        将visual_path收集的all_nodes和all_edges处理为自然语言描述
        
        Args:
            all_nodes: 各层的节点信息列表
            all_edges: 各层的边信息列表
        
        Returns:
            desc: 自然语言格式的子图描述
        """
        # 收集所有层的唯一节点和边
        unique_nodes = set()
        edge_triples = []
        
        # 遍历所有层的edges，收集三元组
        for layer_idx, layer_edges in enumerate(all_edges):
            for edge in layer_edges:
                batch_idx, head_id, rel_id, tail_id = edge[0], edge[1], edge[2], edge[3]
                
                # 跳过自环边
                if rel_id == 2 * self.n_rel:  # self_loop
                    continue
                    
                # 获取实体和关系名称
                head_name = self.loader.id2entity.get(int(head_id), f"entity_{int(head_id)}")
                tail_name = self.loader.id2entity.get(int(tail_id), f"entity_{int(tail_id)}")
                
                unique_nodes.add(head_name)
                unique_nodes.add(tail_name)
                
                # 处理反向关系
                if rel_id >= self.n_rel:
                    rel_name = self.loader.id2relation.get(int(rel_id) - self.n_rel, f"relation_{int(rel_id) - self.n_rel}")
                    # 反向关系：构建自然语言描述
                    triple = (tail_name, rel_name, head_name)
                else:
                    rel_name = self.loader.id2relation.get(int(rel_id), f"relation_{int(rel_id)}")
                    triple = (head_name, rel_name, tail_name)
                
                edge_triples.append(triple)
        
        # 去重并排序
        unique_triples = sorted(set(edge_triples))
        unique_nodes = sorted(unique_nodes)
        
        # 生成自然语言描述
        nl_descriptions = []
        
        # 1. 描述涉及的实体
        if unique_nodes:
            nl_descriptions.append(f"The knowledge graph contains {len(unique_nodes)} entities: {', '.join(unique_nodes[:10])}" + 
                                  (f" and {len(unique_nodes)-10} more" if len(unique_nodes) > 10 else ""))
        
        # 2. 描述关系和连接
        nl_descriptions.append(f"\nThere are {len(unique_triples)} relationships in the graph:")
        
        # 3. 将三元组转换为自然语言句子
        for i, (subj, rel, obj) in enumerate(unique_triples[:20]):  # 限制最多显示20个关系
            # 根据关系类型生成更自然的语言
            if "is" in rel.lower() or "was" in rel.lower():
                sentence = f"- {subj} {rel} {obj}"
            elif "of" in rel.lower() or "in" in rel.lower() or "at" in rel.lower():
                sentence = f"- {subj} has {rel}: {obj}"
            elif rel.endswith("ed") or rel.endswith("ing"):
                sentence = f"- {subj} {rel} {obj}"
            else:
                # 默认格式
                sentence = f"- {subj} has relation '{rel}' with {obj}"
            
            nl_descriptions.append(sentence)
        
        if len(unique_triples) > 20:
            nl_descriptions.append(f"... and {len(unique_triples)-20} more relationships")
        
        # 4. 添加图结构总结
        nl_descriptions.append(f"\nGraph summary: This subgraph explores {len(all_edges)} layers of connections, "
                             f"starting from the query entities and expanding through relevant relationships.")
        
        # 组合所有描述
        desc = "\n".join(nl_descriptions)
        
        return desc

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
