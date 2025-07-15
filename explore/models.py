import torch
import torch.nn as nn
from utils.drug_source import variadic_topk
from torch_scatter import scatter
import numpy as np

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, use_lama_rel, K, sample_flag, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_lama_rel = use_lama_rel
        self.K = K
        self.sample_flag = sample_flag
        
        # 双子图存储
        self.full_subgraph_A = None
        self.attention_subgraph_B = None

        self.Ws_attn = nn.Linear(in_dim, attn_dim)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wq_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.w_alpha  = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, q_emb, rela_embed, hidden, edges, nodes, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx] # q_rel 代表问题的id
        l1 = edges.shape[0]
        n1 = nodes.size(0)
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]
        # print(edges.shape[0])  
        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel,:]
        else:
            hr = rela_embed(rel)
        
        self.n_rel = (rela_embed.shape[0]-1) // 2
        
        r_idx = edges[:,0]
        h_qr = q_emb[edges[:,0],:]
        
        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr)+ self.Wqr_attn(hr * h_qr))))
        
        # 双子图逻辑：先选Top-5存入全局子图A，再选Top-3存入全局子图B
        current_edges_A = None
        current_edges_B = None
        current_nodes_A = None
        current_nodes_B = None
        
        sample_flag = self.sample_flag
        # ========= 双子图逻辑 =============
        if sample_flag == 1 :
            # 第一步：选择Top-5边存入子图A
            max_ent_per_ent_A = 5  # 子图A的Top-5
            _, ind1 = torch.unique(edges[:,0:2],dim=0, sorted=True,return_inverse=True)
            _, ind2 = torch.sort(ind1)
            edges_sorted = edges[ind2]             # sort edges
            alpha_sorted = alpha[ind2]
            _, counts = torch.unique(edges_sorted[:,0:2], dim=0, return_counts=True)
            
            idd_idx = edges_sorted[:,2] == (self.n_rel*2)
            idd_edges = edges_sorted[idd_idx]

            probs = alpha_sorted.squeeze()
            # 选择Top-5存入子图A
            topk_value_A, topk_index_A = variadic_topk(probs, counts, k=max_ent_per_ent_A)
            
            cnt_sum = torch.cumsum(counts,dim=0)
            cnt_sum[1:] = cnt_sum[:-1] + 0
            cnt_sum[0] = 0
            topk_index_A = topk_index_A + cnt_sum.unsqueeze(1)
            
            mask_A = topk_index_A.view(-1,1).squeeze()
            mask_A = torch.unique(mask_A)
            
            # 子图A：Top-5边
            edges_A = edges_sorted[mask_A]
            edges_A = torch.cat((edges_A,idd_edges),0)
            edges_A = torch.unique(edges_A[:,:],dim = 0)
            alpha_A = alpha_sorted[mask_A]
            
            # 当前层的Top-5边存入全局子图A
            current_edges_A = edges_A
            current_alpha_A = alpha_A
            # 提取子图A的节点
            current_nodes_A = torch.unique(edges_A[:,[1,3]], dim=0)  # [head, tail]
            
            # 第二步：从Top-5中选择Top-3边存入全局子图B
            max_ent_per_ent_B = 3  # 子图B的Top-3
            topk_value_B, topk_index_B = variadic_topk(probs, counts, k=max_ent_per_ent_B)
            topk_index_B = topk_index_B + cnt_sum.unsqueeze(1)
            
            mask_B = topk_index_B.view(-1,1).squeeze()
            mask_B = torch.unique(mask_B)
            
            # 当前层的Top-3边存入全局子图B
            current_edges_B = edges_sorted[mask_B]
            current_edges_B = torch.cat((current_edges_B,idd_edges),0)
            current_edges_B = torch.unique(current_edges_B[:,:],dim = 0)
            current_alpha_B = alpha_sorted[mask_B]
            # 提取子图B的节点
            current_nodes_B = torch.unique(current_edges_B[:,[1,3]], dim=0)  # [head, tail]
            
            # 用于实际GNN消息传递的边（当前层的Top-3）
            edges = current_edges_B
            alpha = current_alpha_B

            nodes, tail_index = torch.unique(edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)
            edges = torch.cat([edges[:,0:5], tail_index.unsqueeze(1)], 1)
            
            head_index = edges[:,4]
            idd_mask = edges[:,2] == (self.n_rel*2)
            _, old_idx = head_index[idd_mask].sort()
            old_nodes_new_idx = tail_index[idd_mask][old_idx]

        else:
            pass
            
        # 存储当前层的双子图信息（供全局累积使用）
        self.current_edges_A = current_edges_A
        self.current_edges_B = current_edges_B
        self.current_nodes_A = current_nodes_A
        self.current_nodes_B = current_nodes_B
        
        sub = edges[:,4]
        rel = edges[:,2]
        obj = edges[:,5]
        # print(edges.shape[0])  
        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel,:]
        else:
            hr = rela_embed(rel)
        
        r_idx = edges[:,0]
        h_qr = q_emb[edges[:,0],:]
        
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_qr)+ self.Wqr_attn(hr * h_qr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.size(0), reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))
        # print(nodes.shape, hidden_new.shape)
        l2 = edges.shape[0]
        n2 = nodes.size(0)
        num_node = np.array([n1*1.0/len(q_sub), n2*1.0/len(q_sub)])
        num_edge = np.array([l1*1.0/len(q_sub), l2*1.0/len(q_sub)])

        return num_node, num_edge, hidden_new, alpha, nodes, edges, old_nodes_new_idx

class Explore(torch.nn.Module):
    def __init__(self, params, loader):
        super(Explore, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]
        self.K = params.K
        self.sample_flag = params.sample
        
        # 全局共享的双子图（所有层共用）
        self.global_subgraph_A = {'edges': [], 'nodes': [], 'alpha': []}  # 全局子图A：累积Top-5
        self.global_subgraph_B = {'edges': [], 'nodes': [], 'alpha': []}  # 全局子图B：累积Top-3

        self.question_emb = self.load_qemb().detach() 
        #self.W_q = nn.Linear(5120,self.hidden_dim)
        self.dim_reduct = nn.Sequential(
            nn.Linear(4096, 2096),
            nn.ReLU(),
            nn.Linear(2096, self.hidden_dim)
        ).cuda()

        self.use_lama_rel = 1
        if self.use_lama_rel == 1:
            self.rela_embed = self.load_rel_emb().detach()
        else:
            self.rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)

        self.gnn_layers = []
        for i in range(3):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, self.use_lama_rel, self.K, self.sample_flag, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        
        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)        
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.Wq_final = nn.Linear(self.hidden_dim*2, 1, bias = False)

        self.mlp = nn.Sequential(
            nn.Linear(2*self.hidden_dim, 2*self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2*self.hidden_dim, 1)
        ).cuda()
        
        # 步骤2: 结构向量投影器 (h_g → 软提示向量 ∈ ℝ^{d_LLM})
        self.structure_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),  # Llama2-7b的隐藏维度
        ).cuda()
        self.Wr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)  # r^-1 = Wr+b
        self.loop = nn.Parameter(torch.randn(1, self.hidden_dim))

    def forward(self, subs, qids, mode='train', question_texts=None):
        n_qs = len(qids)
        q_sub = subs
        q_id = torch.LongTensor(qids)# .cuda()

        ques_emb = self.question_emb[q_id,:]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        ques_emb.cpu()
        
        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed)   
            self.rela_embed.cpu()

            rel_emb = rel_emb[0:self.n_rel,:]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop],dim=0)

        else:
            rel_emb = self.rela_embed

        n_node = sum(len(sublist) for sublist in subs)
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),
            np.concatenate(subs)
        ]).reshape(2,-1)
        nodes = np.array(nodes, dtype=np.int64)
        nodes = torch.LongTensor(nodes).T.cuda()

        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        # nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        # hq init hs
        hidden = q_emb[nodes[:,0],:]

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        scores_all = []
        
        # 初始化全局共享的双子图
        self.global_subgraph_A = {'edges': [], 'nodes': [], 'alpha': []}
        self.global_subgraph_B = {'edges': [], 'nodes': [], 'alpha': []}
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids)
            num_node, num_edge, hidden, alpha, nodes, edges, old_nodes_new_idx = self.gnn_layers[i](q_sub, q_id, q_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)
            
            # 将当前层的边和节点累积到全局子图中
            if hasattr(self.gnn_layers[i], 'current_edges_A') and self.gnn_layers[i].current_edges_A is not None:
                self.global_subgraph_A['edges'].append(self.gnn_layers[i].current_edges_A)
                if hasattr(self.gnn_layers[i], 'current_nodes_A') and self.gnn_layers[i].current_nodes_A is not None:
                    self.global_subgraph_A['nodes'].append(self.gnn_layers[i].current_nodes_A)
            if hasattr(self.gnn_layers[i], 'current_edges_B') and self.gnn_layers[i].current_edges_B is not None:
                self.global_subgraph_B['edges'].append(self.gnn_layers[i].current_edges_B)
                if hasattr(self.gnn_layers[i], 'current_nodes_B') and self.gnn_layers[i].current_nodes_B is not None:
                    self.global_subgraph_B['nodes'].append(self.gnn_layers[i].current_nodes_B)
            
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate (hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            num_nodes[i,:] += num_node
            num_edges[i,:] += num_edge

        h_qs = q_emb[nodes[:,0],:]
        scores = self.mlp(torch.cat((hidden,h_qs),dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()         
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        # 步骤2: 生成结构向量 h_g
        h_g = self.generate_structure_vector(hidden, nodes)
        
        # 步骤3: 生成文本向量 h_t
        h_t = None
        if question_texts is not None and len(question_texts) > 0:
            h_t = self.generate_text_vector(question_texts[0])
        
        # 步骤4: 候选实体与推理链生成
        multi_choice_prompt = self.generate_candidates_and_chains(scores_all)
        
        # 重新生成包含多选提示的文本向量
        if question_texts is not None and len(question_texts) > 0:
            h_t = self.generate_text_vector(question_texts[0], multi_choice_prompt)
        
        if mode == 'train':
            return num_nodes, num_edges, scores_all, h_g, h_t, multi_choice_prompt     
        else:
            return scores_all, h_g, h_t, multi_choice_prompt
    
    def get_global_subgraph_A(self):
        """获取合并后的全局子图A（所有层的Top-5边）"""
        if len(self.global_subgraph_A['edges']) == 0:
            return None
        all_edges = torch.cat(self.global_subgraph_A['edges'], dim=0)
        unique_edges = torch.unique(all_edges, dim=0)
        return unique_edges
    
    def get_global_subgraph_B(self):
        """获取合并后的全局子图B（所有层的Top-3边）"""
        if len(self.global_subgraph_B['edges']) == 0:
            return None
        all_edges = torch.cat(self.global_subgraph_B['edges'], dim=0)
        unique_edges = torch.unique(all_edges, dim=0)
        return unique_edges
    
    def get_global_subgraph_A_with_nodes(self):
        """获取合并后的全局子图A（边和节点）"""
        if len(self.global_subgraph_A['edges']) == 0:
            return None, None
        
        # 合并所有边
        all_edges = torch.cat(self.global_subgraph_A['edges'], dim=0)
        unique_edges = torch.unique(all_edges, dim=0)
        
        # 合并所有节点
        unique_nodes = None
        if len(self.global_subgraph_A['nodes']) > 0:
            all_nodes = torch.cat(self.global_subgraph_A['nodes'], dim=0)
            unique_nodes = torch.unique(all_nodes, dim=0)
        
        return unique_edges, unique_nodes
    
    def get_global_subgraph_B_with_nodes(self):
        """获取合并后的全局子图B（边和节点）"""
        if len(self.global_subgraph_B['edges']) == 0:
            return None, None
        
        # 合并所有边
        all_edges = torch.cat(self.global_subgraph_B['edges'], dim=0)
        unique_edges = torch.unique(all_edges, dim=0)
        
        # 合并所有节点
        unique_nodes = None
        if len(self.global_subgraph_B['nodes']) > 0:
            all_nodes = torch.cat(self.global_subgraph_B['nodes'], dim=0)
            unique_nodes = torch.unique(all_nodes, dim=0)
        
        return unique_edges, unique_nodes
    
    def generate_structure_vector(self, hidden, nodes):
        """
        步骤2: 结构向量生成
        对全局子图B的节点做均值池化，然后用MLP投影到LLM维度
        
        Args:
            hidden: 当前节点的隐藏表示 [n_nodes, hidden_dim]
            nodes: 当前节点的索引 [n_nodes, 2] (batch_idx, node_idx)
        
        Returns:
            h_g_projected: 投影后的结构向量 [batch_size, 4096]
        """
        
        # 获取全局子图B的节点
        edges_B, nodes_B = self.get_global_subgraph_B_with_nodes()
        
        if nodes_B is None or len(self.global_subgraph_B['edges']) == 0:
            # 如果没有子图B，使用当前所有节点的均值
            h_g = torch.mean(hidden, dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            # 使用全局子图B中的节点做均值池化
            # nodes_B格式: [node_pairs, 2] 包含 (head, tail)
            unique_node_ids = torch.unique(nodes_B.flatten())  # 获取所有唯一节点ID
            
            # 从当前节点中找到对应的节点表示
            node_embeddings = []
            for node_id in unique_node_ids:
                # 在nodes中找到该node_id的位置
                mask = (nodes[:, 1] == node_id)
                if mask.any():
                    node_embed = hidden[mask][0]  # 取第一个匹配的表示
                    node_embeddings.append(node_embed)
            
            if len(node_embeddings) > 0:
                # 对全局子图B的节点做均值池化
                h_g = torch.stack(node_embeddings).mean(dim=0, keepdim=True)  # [1, hidden_dim]
            else:
                # 如果没有找到对应节点，使用全部节点均值
                h_g = torch.mean(hidden, dim=0, keepdim=True)
        
        # 用多层MLP投影到LLM维度
        h_g_projected = self.structure_projector(h_g)  # [1, 4096]
        
        # 扩展到batch维度
        batch_size = torch.unique(nodes[:, 0]).size(0)
        h_g_projected = h_g_projected.expand(batch_size, -1)  # [batch_size, 4096]
        
        return h_g_projected
    
    def textualize_subgraph(self, question_text):
        """
        步骤3: 文本化子图
        将全局子图A转换为CSV格式的desc形式
        
        Args:
            question_text: 问题文本
        
        Returns:
            desc: CSV格式的子图描述 + 问题
        """
        # 获取全局子图A（完整子图）
        edges_A = self.get_global_subgraph_A()
        
        if edges_A is None:
            return f"Question: {question_text}"
        
        # 收集所有唯一节点和边
        unique_nodes = set()
        edge_data = []
        
        for edge in edges_A:
            batch_idx, head_id, rel_id, tail_id = edge[0].item(), edge[1].item(), edge[2].item(), edge[3].item()
            
            # 跳过自环边
            if rel_id == 2 * self.n_rel:  # self_loop
                continue
                
            # 获取实体和关系名称
            head_name = self.loader.id2entity.get(head_id, f"entity_{head_id}")
            tail_name = self.loader.id2entity.get(tail_id, f"entity_{tail_id}")
            
            unique_nodes.add((head_id, head_name))
            unique_nodes.add((tail_id, tail_name))
            
            # 处理反向关系
            if rel_id >= self.n_rel:
                rel_name = self.loader.id2relation.get(rel_id - self.n_rel, f"relation_{rel_id - self.n_rel}")
                # 反向关系：交换head和tail
                edge_data.append((tail_name, rel_name, head_name))
            else:
                rel_name = self.loader.id2relation.get(rel_id, f"relation_{rel_id}")
                edge_data.append((head_name, rel_name, tail_name))
        
        # 生成节点CSV
        node_csv_lines = ["node_name"]
        for node_id, node_name in sorted(unique_nodes, key=lambda x: x[1]):
            node_csv_lines.append(node_name)
        nodes_csv = "\n".join(node_csv_lines)
        
        # 生成边CSV
        edge_csv_lines = ["src,edge_attr,dst"]
        for src, rel, dst in sorted(set(edge_data)):  # 去重
            # 转义CSV中的逗号和引号
            src_escaped = f'"{src}"' if ',' in src else src
            rel_escaped = f'"{rel}"' if ',' in rel else rel
            dst_escaped = f'"{dst}"' if ',' in dst else dst
            edge_csv_lines.append(f"{src_escaped},{rel_escaped},{dst_escaped}")
        edges_csv = "\n".join(edge_csv_lines)
        
        # 按照desc格式组合：nodes + edges
        desc = f"{nodes_csv}\n\n{edges_csv}"
        
        return desc
    
    def generate_text_vector(self, question_text, multi_choice_prompt=None, max_seq_len=512):
        """
        步骤3: 文本向量生成
        多选提示 + 全局子图A文本化(CSV格式) + 问题 → LLM Token Embedding → h_t
        """
        # 获取CSV格式的子图描述
        desc = self.textualize_subgraph(question_text)
        
        # 构建完整的文本输入，按照流程.txt中的格式
        # 格式：多选提示 + 图结构(CSV) + Question: + 问题
        if multi_choice_prompt:
            full_text = f"多选提示：{multi_choice_prompt}\n图结构:{desc}\nQuestion:{question_text}"
        else:
            full_text = f"图结构:\n{desc}\nQuestion:\n{question_text}"
        
        # 懒加载LLM
        if not hasattr(self, 'llm_model'):
            self._load_llm_model()
        
        # 获取Token Embedding
        inputs = self.llm_tokenizer(
            full_text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding=False
        )
        
        with torch.no_grad():
            input_ids = inputs["input_ids"].to(self.llm_model.device)
            h_t = self.llm_model.model.embed_tokens(input_ids).squeeze(0)  # [L, 4096]
        
        return h_t
    
    def _load_llm_model(self):
        """懒加载Llama2-7b-chat-hf，int8量化"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # 冻结参数避免 CPU↔GPU 切换
            for param in self.llm_model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"LLM加载失败: {e}")
            self.llm_model = None
    
    def generate_candidates_and_chains(self, scores_all, top_n=3):
        """
        步骤4: 候选实体与推理链生成
        用GNN logits选Top-N候选实体，回溯推理链
        
        Args:
            scores_all: GNN输出分数 [batch_size, n_entities]
            top_n: 候选实体数量
        
        Returns:
            multi_choice_prompt: 多选提示字符串
        """
        # 获取Top-N候选实体
        batch_size = scores_all.size(0)
        candidates = []
        
        for batch_idx in range(batch_size):
            scores = scores_all[batch_idx]
            
            # 软最大化获取概率
            probs = torch.softmax(scores, dim=0)
            
            # Top-N候选
            top_probs, top_indices = torch.topk(probs, top_n)
            
            batch_candidates = []
            for i, (entity_id, prob) in enumerate(zip(top_indices, top_probs)):
                entity_name = self.loader.id2entity.get(entity_id.item(), f"entity_{entity_id.item()}")
                
                # 回溯推理链
                chain = self._backtrack_reasoning_chain(entity_id.item(), batch_idx)
                
                batch_candidates.append({
                    'entity_id': entity_id.item(),
                    'entity_name': entity_name,
                    'probability': prob.item(),
                    'chain': chain
                })
            
            candidates.append(batch_candidates)
        
        # 构造多选提示
        multi_choice_prompt = self._build_multi_choice_prompt(candidates[0] if candidates else [])
        
        return multi_choice_prompt
    
    def _backtrack_reasoning_chain(self, target_entity_id, batch_idx):
        """回溯推理链：从第3跳向第1跳回溯α最大边"""
        if len(self.global_subgraph_B['edges']) == 0:
            return "No reasoning chain available."
        
        # 简化实现：找到目标实体的相关边
        chain_parts = []
        edges_B = self.get_global_subgraph_B()
        
        if edges_B is not None:
            # 找到与目标实体相关的边
            target_edges = []
            for edge in edges_B:
                if edge[3].item() == target_entity_id:  # tail == target
                    target_edges.append(edge)
            
            # 取前几条边作为推理链
            for i, edge in enumerate(target_edges[:2]):  # 最多2条边
                head_id, rel_id, tail_id = edge[1].item(), edge[2].item(), edge[3].item()
                head_name = self.loader.id2entity.get(head_id, f"entity_{head_id}")
                tail_name = self.loader.id2entity.get(tail_id, f"entity_{tail_id}")
                
                if rel_id < self.n_rel:
                    rel_name = self.loader.id2relation.get(rel_id, f"relation_{rel_id}")
                    chain_parts.append(f"{head_name} → {rel_name} → {tail_name}")
        
        return " ; ".join(chain_parts) if chain_parts else "Direct answer"
    
    def _build_multi_choice_prompt(self, candidates):
        """构造多选提示"""
        if not candidates:
            return "No candidates available."
        
        labels = ['A', 'B', 'C', 'D', 'E']
        prompt_parts = []
        
        for i, candidate in enumerate(candidates):
            if i >= len(labels):
                break
            
            label = labels[i]
            entity_name = candidate['entity_name']
            prob = candidate['probability']
            chain = candidate['chain']
            
            prompt_parts.append(f"{label}. {entity_name} (p={prob:.3f}) {{{chain}}}")
        
        return "\n".join(prompt_parts)
    

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
            q_train = np.load('../embedding/webqsp-train.npy') 
            q_valid = np.load('../embedding/webqsp-valid.npy')
            q_test = np.load('../embedding/webqsp-test.npy') 
        elif 'CWQ' in datapath:
            q_train = np.load('../embedding/CWQ-train.npy') 
            q_valid = np.load('../embedding/CWQ-valid.npy')
            q_test = np.load('../embedding/CWQ-test.npy') 

        q_emb = np.concatenate((q_train,q_valid,q_test))

        return torch.tensor(q_emb,dtype=torch.float32)
    
    def load_rel_emb(self):

        datapath = self.loader.task_dir
        if 'MetaQA' in datapath:
            rel_emb = np.load('../embedding/Meta-rel.npy')
        elif 'webqsp' in datapath:
            rel_emb = np.load('../embedding/webqsp-rel.npy')
        elif 'CWQ' in datapath:
            rel_emb = np.load('../embedding/CWQ-rel.npy')

        print('rel_emb shape: ',rel_emb.shape) 

        return torch.tensor(rel_emb,dtype=torch.float32)
    
    def change_loader(self, loader):

        self.loader = loader
        self.question_emb = self.load_qemb().detach()
        self.rela_embed = self.load_rel_emb().detach()
        self.n_rel = self.loader.n_rel
        print('change loader:', self.loader.task_dir)


    def visual_path(self, subs, qids, objs, filepath, mode='test'):

        n_qs = len(qids)
        q_sub = subs
        q_id = torch.LongTensor(qids)
        
        ques_emb = self.question_emb[q_id,:]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        ques_emb.cpu()
        
        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed) 
            self.rela_embed.cpu()

            rel_emb = rel_emb[0:self.n_rel,:]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop],dim=0)
        else:
            rel_emb = self.rela_embed

        n_node = sum(len(sublist) for sublist in subs)
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),
            np.concatenate(subs)
        ]).reshape(2,-1)
        nodes = np.array(nodes, dtype=np.int64)
        nodes = torch.LongTensor(nodes).T.cuda()

        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        # nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        hidden = q_emb[nodes[:,0],:]

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))

        all_nodes = []
        all_edges = []
        all_weights = []
        min_weight = []

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx  = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids)

            num_node, num_edge, hidden, weights, nodes, edges, old_nodes_new_idx = self.gnn_layers[i](q_sub, q_id, q_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)
            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate (hidden.unsqueeze(0), h0)  #  
            hidden = hidden.squeeze(0)
            # print(i,torch.max(weights),torch.min(weights))
            all_nodes.append(nodes.cpu().data.numpy())
            all_edges.append(edges.cpu().data.numpy())
            all_weights.append(weights.cpu().data.numpy())
            min_weight.append(torch.min(weights).item())

        h_qs = q_emb[nodes[:,0],:]
        scores = self.mlp(torch.cat((hidden,h_qs),dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()       
        scores_all[[nodes[:,0], nodes[:,1]]] = scores
        scores_all = scores_all.squeeze().cpu().data.numpy()
        n = 10
        top_indices = np.argsort(scores_all)[::-1][:n]
        answer = top_indices

        softscore = self.softmax(scores_all)
        top_values = np.partition(softscore, -2)[::-1][:n]
        probs = top_values

        f = open(filepath,'+a')
        qs = qids - self.loader.n_valid_qs - self.loader.n_train_qs

        f.write(f'{qs[0]}\t')

        for k in range(n):
            tails = answer[k]
            outstr = 'tail: %d,  p:%.2f' % (tails, probs[k])

            f.write('%s|%0.3f|'%(self.loader.id2entity[answer[k]], probs[k]))
            print_edges = []
            for i in range(self.n_layer-1, -1, -1):
                # print('layer:',i)
                edges = all_edges[i]   
                # print(edges.shape)
                weights = all_weights[i]
                mask1 = edges[:,3] == tails 
                if np.sum(mask1) == 0:
                    tails = edges[0,3]
                    mask1 = edges[:,3] == tails
                weights1 = weights[mask1].reshape(-1,1)
                edges1 = edges[mask1]
                mask2 = np.argmax(weights1)
                
                new_edges = edges1[mask2].reshape(1,-1)
                #print(new_edges.shape)
                new_weights = np.round_(weights1[mask2], 2).reshape(-1,1)
                #print(new_weights.shape)
                new_edges = np.concatenate([new_edges[:,[1,2,3]], new_weights], 1)
                # new_edges: [h,r,t,alpha]
                tails = new_edges[:,0].astype('int') 
                print_edges.append(new_edges)
                
    
            for i in range(self.n_layer-1, -1, -1):
                edge = print_edges[i][0].tolist()
                outstr = '%d\t %d\t %d\t%.4f'% (edge[0], edge[1], edge[2], edge[3])

                if edge[1] < self.loader.n_rel:
                    h = self.loader.id2entity[int(edge[0])]
                    r = self.loader.id2relation[int(edge[1])]
                    t = self.loader.id2entity[int(edge[2])]
                    f.write('(' + h + ', ' + r +', ' + t + ');')
                elif edge[1] == 2*self.n_rel:
                    h = self.loader.id2entity[int(edge[0])]
                    r = self.loader.id2relation[int(edge[1])]
                    t = self.loader.id2entity[int(edge[2])]
                    f.write('(' + h + ', ' + r +', ' + t + ');')
                else:
                    h = self.loader.id2entity[int(edge[0])]
                    r = self.loader.id2relation[int(edge[1])-self.loader.n_rel]
                    t = self.loader.id2entity[int(edge[2])]
                    f.write('(' + t + ', ' + r +', ' + h + ');')    
            f.write('\t')
        f.write('\n')            
           
        return True
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)