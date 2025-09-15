import contextlib
import torch
import torch.nn as nn
from utils.utils import candidate_path# 期望你项目内已有该函数
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models import Explore
from load_data import DataLoader
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig
import json
import numpy as np
from pathlib import Path
import os
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100


class GraphLLM(torch.nn.Module):
    """
    统一架构的GraphLLM，融合GNN和LLM进行端到端训练
    按照流程.txt中的设计：
    1. GNN编码图结构
    2. 图嵌入投影到LLM维度
    3. 图嵌入与文本嵌入拼接作为软提示
    4. 计算统一的因果语言建模损失
    """

    def __init__(
        self,
        args,
        loader,
        pretrained_gnn_path=None,
        freeze_gnn=False,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.loader = loader
        # 保存数据集名称用于处理 WebCWQ
        self.dataset_name = getattr(args, 'dataset', None)
        
        # 为 WebCWQ 准备多个 GNN 权重路径
        self.multi_gnn_paths = {}
        if self.dataset_name == 'WebCWQ' and isinstance(pretrained_gnn_path, dict):
            # 如果传入的是字典，格式: {'webqsp': 'path1.pt', 'cwq': 'path2.pt'}
            # 统一键名为小写，避免大小写不一致导致查找失败
            self.multi_gnn_paths = {str(k).lower(): v for k, v in pretrained_gnn_path.items()}
            pretrained_gnn_path = None  # 初始化时不加载，等待动态切换
        elif self.dataset_name == 'WebCWQ' and isinstance(pretrained_gnn_path, str):
            # 如果传入单个路径，尝试自动推断其他路径
            import os
            base_dir = os.path.dirname(pretrained_gnn_path)
            self.multi_gnn_paths = {
                'webqsp': os.path.join(base_dir, 'webqsp_best_saved_model.pt'),
                'cwq': os.path.join(base_dir, 'CWQ_best_saved_model.pt')
            }
            pretrained_gnn_path = None
        else:
            # 非 WebCWQ 的场景，清空多权重表
            self.multi_gnn_paths = {}
        # 当前已加载的 GNN 权重路径（用于避免重复加载）
        self._current_gnn_weights = None
        # 控制候选是否包含 概率 和 facts
        self.include_prob_facts = bool(getattr(args, 'include_prob_facts', True))
        # 控制是否使用GNN soft prompt (h_g)
        self.use_graph_prompt = bool(getattr(args, 'use_graph_prompt', True))  # 默认启用

        print('Loading LLAMA')
        
        # 8bit量化配置
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        # 设备映射：默认 auto；若强制单卡或只有单卡，则绑定到 cuda:0
        single_card = bool(getattr(args, 'force_single_card', False)) or (torch.cuda.device_count() <= 1)
        
        # 修复：使用字符串格式的设备映射以确保正确的设备分配
        if single_card:
            device_map = {"": "cuda:0"}  # 映射所有层到 cuda:0
        else:
            device_map = "auto"
            
        llm_kwargs = {
            "max_memory": {0: '40GiB', "cpu": "30GiB"},  # 添加CPU内存限制以防止过度使用
            "device_map": device_map,
            "revision": "main",
            "quantization_config": quant_config,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=llm_kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            **llm_kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
            # 确保在仅训练投影器时，LLM 仍对 inputs_embeds 建立梯度，便于梯度回传到 h_g_projector
            try:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                if hasattr(model, 'enable_input_require_grads'):
                    model.enable_input_require_grads()
                if hasattr(model, 'config'):
                    model.config.use_cache = False
                print("Enabled input grads and disabled cache for frozen LLM.")
            except Exception as e:
                print(f"Warning enabling input grads/ckpt: {e}")
        else:
            print("Training LLAMA with LORA!")
            # Prepare model for k-bit training
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 16  # 增大秩以捕获更复杂的KGQA模式
            lora_alpha: int = 32  # 通常设为 2 * lora_r
            lora_dropout: float = 0.05  # 保持较低的dropout以维持精确性
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        # 训练/计算 loss 时关闭 cache（LoRA/常规训练也受益）
        try:
            model.config.use_cache = False
        except Exception:
            pass
        self.model = model
        print('Finish loading LLAMA!')

        # 集成现有的GNN模型（Explore类）
        # 确保与GraphLLM使用相同的设备
        graph_device = self.model.device if hasattr(self.model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_encoder = Explore(args, loader, device=graph_device)
        
        # h_g投影器：将GNN输出的h_g_pooled投影到LLM维度
        # h_g_pooled: [batch, hidden_dim] -> h_g: [batch, 5120]
        # 使用三层映射，逐步扩展维度，避免维度跳跃过大
        self.h_g_projector = nn.Sequential(
            nn.Linear(256, 768),      # 256 -> 768 (3x)
            nn.ReLU(),
            nn.Dropout(0.1),          # 添加 dropout 防止过拟合
            nn.Linear(768, 2048),      # 768 -> 2048 (2.67x)
            nn.ReLU(),
            nn.Dropout(0.1),          # 添加 dropout
            nn.Linear(2048, 5120)      # 2048 -> 5120 (2.5x)
        ).to(self.model.device)
        

        
        # 加载预训练的GNN权重
        if pretrained_gnn_path:
            self.load_pretrained_gnn(pretrained_gnn_path)
        
        # 冻结GNN参数（现在h_g_projector和soft_prompt_projector已经创建）
        if freeze_gnn:
            self.freeze_gnn_parameters()

        self.word_embedding = self.model.model.get_input_embeddings()
        
        # 预加载所有path文件到内存，避免训练时的I/O开销
        self.preload_all_paths()
        # 缓存已分词的提示输入（按数据集划分与相对 qid）
        if not hasattr(self, "_ht_tok_cache"):
            self._ht_tok_cache = {}
        # 缓存图侧向量（h_g_pooled，未过 projector），仅当 GNN 冻结时启用
        if not hasattr(self, "_hg_cache"):
            self._hg_cache = {}
        # 是否允许缓存图向量（GNN 不训练时安全）
        try:
            self._can_cache_hg = not any(p.requires_grad for p in self.graph_encoder.parameters())
        except Exception:
            self._can_cache_hg = False

    def _resolve_path_file(self, dataset_short: str, split: str):
        """根据数据集缩写和划分名推断可用的 *-path.txt 文件路径。
        优先顺序：explore/13b-path -> explore/ -> 单文件后备。
        自动处理 dev/valid 命名差异与大小写。
        返回存在的路径字符串，若均不可用则返回 None。
        """
        ds_candidates = [dataset_short.lower(), dataset_short.upper()]
        split_aliases = [split]
        if split == 'dev':
            split_aliases.append('valid')
        # 相对本文件与相对 CWD 的多层候选
        here = os.path.dirname(__file__)
        candidates = []
        for ds in ds_candidates:
            for sp in split_aliases:
                candidates.extend([
                    os.path.join(here, '13b-path', f'{ds}-{sp}-path.txt'),
                    os.path.join(here, f'{ds}-{sp}-path.txt'),
                    os.path.join('explore', '13b-path', f'{ds}-{sp}-path.txt'),
                    os.path.join('explore', f'{ds}-{sp}-path.txt'),
                ])
            # 单文件兜底（无 split）
            candidates.extend([
                os.path.join(here, f'{ds}-path.txt'),
                os.path.join('explore', f'{ds}-path.txt'),
            ])
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _split_info(self, qid: int):
        """返回 (split, relative_qid, split_file) 供缓存与 prompt 构造使用。"""
        if isinstance(qid, torch.Tensor):
            qid = int(qid.item())
        else:
            qid = int(qid)
        
        # 对于 WebCWQ，需要根据当前 loader 的 task_dir 来判断
        # WebCWQ 训练时会交替使用 webqsp_nsm 和 CWQ 的 loader
        task = self.loader.task_dir.lower()
        ds_tag = 'webqsp' if 'webqsp' in task else ('cwq' if 'cwq' in task else 'metaqa')
        if qid < self.loader.n_train_qs:
            split = 'train'
            rel_qid = qid
            split_file = self._resolve_path_file(ds_tag, 'train')
        elif qid < self.loader.n_train_qs + self.loader.n_valid_qs:
            split = 'valid'
            rel_qid = qid - self.loader.n_train_qs
            # 兼容 dev/valid 命名
            split_file = self._resolve_path_file(ds_tag, 'dev') or self._resolve_path_file(ds_tag, 'valid')
        else:
            split = 'test'
            rel_qid = qid - self.loader.n_train_qs - self.loader.n_valid_qs
            split_file = self._resolve_path_file(ds_tag, 'test')
        
        return split, rel_qid, split_file

    def switch_gnn_weights(self, dataset_key):
        """根据数据集动态切换 GNN 权重"""
        if not self.multi_gnn_paths:
            return  # 没有多个权重路径，不需要切换
        
        # 确定要加载的权重（统一使用小写键）
        key = None
        dkl = dataset_key.lower()
        if 'webqsp' in dkl:
            key = 'webqsp'
        elif 'cwq' in dkl:
            key = 'cwq'

        gnn_path = self.multi_gnn_paths.get(key) if key else None
        
        if gnn_path and hasattr(self, '_current_gnn_weights'):
            # 如果已经是相同的权重，不重复加载
            if self._current_gnn_weights == gnn_path:
                return
        
        if gnn_path:
            import os
            if os.path.exists(gnn_path):
                print(f"Switching GNN weights to: {gnn_path}")
                self.load_pretrained_gnn(gnn_path)
                self._current_gnn_weights = gnn_path
                # 清空缓存，因为切换了权重
                self._hg_cache.clear()
                print(f"Cleared h_g cache after switching weights")
            else:
                print(f"Warning: GNN weight file not found: {gnn_path}")
    
    def _get_hg_pooled_cached(self, subs, qids, mode='llm_train'):
        """按 (split, rel_qid) 缓存/获取 h_g_pooled（GNN 输出；未过 projector）。
        仅当 GNN 冻结时启用缓存；否则现算。
        对于 WebCWQ，会根据当前数据集动态切换 GNN 权重。
        返回: h_g_pooled [B, hidden_dim]
        """
        # WebCWQ: 动态切换 GNN 权重
        if self.dataset_name == 'WebCWQ' and self.multi_gnn_paths:
            # 根据当前 loader 的 task_dir 切换权重
            task = self.loader.task_dir.lower()
            self.switch_gnn_weights(task)
        
        B = len(qids)
        device = self.graph_encoder.device
        hidden_dim = getattr(self.graph_encoder, 'hidden_dim', 256)
        h_g_pooled_out = torch.zeros(B, hidden_dim, dtype=torch.float32, device=device)

        # 收集未命中项
        miss_idx = []
        miss_subs = []
        miss_qids = []
        keys = []
        for i in range(B):
            split, rel_qid, _ = self._split_info(int(qids[i]))
            key = (split, int(rel_qid))
            keys.append(key)
            if self._can_cache_hg and key in self._hg_cache:
                h_cached = self._hg_cache[key]
                h_g_pooled_out[i] = h_cached.to(device)
            else:
                miss_idx.append(i)
                miss_subs.append(subs[i])
                miss_qids.append(int(qids[i]))

        if len(miss_idx) > 0:
            # 现算未命中项
            results = self.graph_encoder(miss_subs, miss_qids, mode=mode)
            if len(results) == 3:
                h_g_pooled_miss, _, _ = results
            else:
                h_g_pooled_miss, _ = results
            # 回填并写缓存
            for j, i in enumerate(miss_idx):
                vec = h_g_pooled_miss[j]
                h_g_pooled_out[i] = vec
                if self._can_cache_hg:
                    self._hg_cache[keys[i]] = vec.detach().cpu()

        return h_g_pooled_out

    def _build_chat_text(self, question_text, multi_choice_prompt=None):
        """构造对话模板文本并追加 'Answer: ' 前缀（仅返回字符串）。"""
        # 与 generate_text_vector 的系统提示语义一致，稍微压缩以减少长度
        system_prompt = (
            "You are a KGQA expert. You will be given a question and candidates A/B/C/... .\n\n"
            "Decision Process:\n"
            "1) Understand the question carefully.\n"
            "2) Evaluate candidates using your knowledge.\n"
            "3) If one candidate is CORRECT, choose it.\n"
            "4) If none are correct, ignore candidates and answer from your knowledge.\n\n"
            "Output Rules:\n"
            "- Start with: Answer: <...>\n"
            "- If selecting a candidate: output 'A. <candidate>' exactly.\n"
            "- If rejecting all candidates: output your own answer text only.\n"
            "- Do NOT output explanations or chain-of-thought."
        )
        if multi_choice_prompt:
            user_content = (
                "Candidates with supporting evidence (evaluate correctness):\n"
                f"{multi_choice_prompt}\n\n"
                f"Question: {question_text}\n\nAnswer:"
            )
        else:
            user_content = f"Question: {question_text}\n\nAnswer:"

        formatted_prompt = (
            "<s>[INST] <<SYS>>\n"
            f"{system_prompt}\n"
            "<</SYS>>\n\n"
            f"{user_content} [/INST]"
        )
        return formatted_prompt

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def preload_all_paths(self):
        """预加载所有path文件到内存缓存"""
        print("Preloading all path files to memory...")
        if not hasattr(self, "_cand_cache"):
            self._cand_cache = {}
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache = {}
        
        # 根据数据集类型确定要加载的文件（使用解析器选择可用文件）
        path_files = []
        if self.dataset_name == 'WebCWQ':
            print("  Loading WebCWQ: will attempt to load both webqsp and CWQ path files")
            for ds in ['webqsp', 'cwq']:
                for sp in ['train', 'dev', 'valid', 'test']:
                    p = self._resolve_path_file(ds, sp)
                    if p and p not in path_files:
                        path_files.append(p)
        else:
            task = self.loader.task_dir.lower()
            if 'webqsp' in task:
                ds = 'webqsp'
            elif 'cwq' in task:
                ds = 'cwq'
            elif 'metaqa' in task:
                ds = 'metaqa'
            else:
                ds = None
            if ds:
                for sp in ['train', 'dev', 'valid', 'test']:
                    p = self._resolve_path_file(ds, sp)
                    if p and p not in path_files:
                        path_files.append(p)

        for filepath in path_files:
            if filepath not in self._cand_cache:
                try:
                    print(f"  Loading {filepath}...")
                    all_candi, all_score, all_p, all_ids = candidate_path(filepath)
                    self._cand_cache[filepath] = (all_candi, all_score, all_p, all_ids)
                    print(f"    Loaded {len(all_ids)} questions from {filepath}")
                except Exception as e:
                    print(f"    Warning: Failed to load {filepath}: {e}")
                    self._cand_cache[filepath] = None
        
        print("Path files preloading completed!")

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            # Use the new torch.amp.autocast API to avoid deprecation warning
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()


    def load_multi_choice_prompt_from_file(self, qid, filepath=None):
        """
        根据 <dataset>-path.txt 构造多选提示串（参考 hintABCpp 形式）并返回。
        形式示例：
          " A. <cand0> (correct probability: <p0>)  {relevant facts: <triples0>}
            B. <cand1> (correct probability: <p1>)  {relevant facts: <triples1>}
            C. <cand2> (correct probability: <p2>)  {relevant facts: <triples2>}  Answer: "

        说明：
        - 优先返回前三个候选（A/B/C）。若不足 3 个则返回现有个数（仅 A，或 A/B）。
        - 当 qid 不在 path 文件中时，返回 " Answer: "（与参考代码在 else 分支一致，由外层拼接 Question）。
        - 结果做缓存，避免重复解析文件。
        """
        # 结果缓存（按 qid + filepath）
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache = {}
        cache_key = (int(qid), str(filepath))
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        # 解析 path 文件缓存（按 filepath）
        if not hasattr(self, "_cand_cache"):
            self._cand_cache = {}
        if filepath is None:
            # 未提供路径文件：返回空 Answer:
            self._prompt_cache[cache_key] = " Answer: "
            return " Answer: "

        if filepath not in self._cand_cache:
            try:

                all_candi, all_score, all_p, all_ids = candidate_path(filepath)
                self._cand_cache[filepath] = (all_candi, all_score, all_p, all_ids)
            except Exception as e:
                print(f"Warning: candidate_path failed for {filepath}: {e}")
                self._cand_cache[filepath] = None

        data = self._cand_cache.get(filepath)
        if not data:
            # 解析失败：返回空的 Answer 提示以保持流程不中断
            self._prompt_cache[cache_key] = " Answer: "
            return " Answer: "

        all_candi, all_score, all_p, all_ids = data

        # 将传入的相对 qid 映射到该题在 all_* 列表中的行号
        qid = int(qid)
        if qid not in all_ids:
            # 与参考代码的 else 分支一致：无候选则只给 Answer:
            self._prompt_cache[cache_key] = " Answer: "
            return " Answer: "

        i = all_ids[qid]

        # 组装 A/B/C：按开关决定是否包含 概率 与 facts
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n_cand = min(3, len(all_candi[i]))
        parts = []
        for k in range(n_cand):
            cand = str(all_candi[i][k]) if k < len(all_candi[i]) else ""
            if self.include_prob_facts:
                # 尽量保持原格式
                try:
                    prob = all_score[i][k]
                except Exception:
                    prob = ""
                try:
                    triples = all_p[i][k]
                except Exception:
                    triples = ""
                part = f"{letters[k]}. {cand} (correct probability: {prob})  {{relevant facts: {triples}}}"
            else:
                part = f"{letters[k]}. {cand}"
            parts.append(part)

        result = "\n".join(parts)
        self._prompt_cache[cache_key] = result
        # print("result",'---------',result)
        return result

    def encode_graphs(self, subs, qids, question_texts=None, mode='llm_train'):
        """
        步骤1: GNN编码图结构 - 批量化版本，一次性处理所有问题
        """
        # 直接调用GNN的批量forward
        results = self.graph_encoder(subs, qids, mode=mode, question_texts=question_texts)
        
        if len(results) == 3:
            h_g_pooled, subgraph, scores = results
        else:
            h_g_pooled, subgraph = results
            scores = None
        
        # 投影到LLM维度
        h_g = self.h_g_projector(h_g_pooled).to(torch.float16)  # 转换为float16
        
        # 如果需要单独的子图列表，这里拆分
        subgraph_list = [subgraph] * len(qids) if not isinstance(subgraph, list) else subgraph
        
        return h_g, scores, subgraph_list


    def forward(self, batch):
        """
        统一前向传播和损失计算 (对应graph_llm.py:171)
        """
        subs = batch["subs"]
        qids = batch["qids"] 
        questions = batch["question"]
        labels = batch["label"]

        # 步骤1: 图编码（带缓存 h_g_pooled；随后过 projector 以参与训练）
        h_g_pooled = self._get_hg_pooled_cached(subs, qids, mode='llm_train')
        h_g = self.h_g_projector(h_g_pooled).to(torch.float16)  # 转换为float16.to(torch.float16)  # [B, H] - 转换为float16

        # 步骤2: 批量生成文本 token 并一次性嵌入
        texts = []
        ids_list, attn_list, meta = [], [], []
        pad_id = int(self.tokenizer.pad_token_id)
        batch_size = len(questions)
        for i, question in enumerate(questions):
            current_qid = qids[i] if isinstance(qids, list) else int(qids[i].item())
            split, relative_qid, split_file = self._split_info(current_qid)

            multi_choice_prompt = self.load_multi_choice_prompt_from_file(relative_qid, split_file)
            cache_key = (split, int(relative_qid))
            if cache_key in self._ht_tok_cache:
                ids, attn = self._ht_tok_cache[cache_key]
                ids_list.append(ids)
                attn_list.append(attn)
                texts.append(None)
            else:
                full_text = self._build_chat_text(question, multi_choice_prompt)
                texts.append(full_text)
                ids_list.append(None)
                attn_list.append(None)
            meta.append(cache_key)

        # 批量分词未缓存的样本（不填充），随后统一手动左填充到同长度
        uncached_idx = [i for i, ids in enumerate(ids_list) if ids is None]
        if len(uncached_idx) > 0:
            to_tok = [texts[i] for i in uncached_idx]
            tok = self.tokenizer(
                to_tok,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=self.max_txt_len,
            )
            for j, i in enumerate(uncached_idx):
                ids = tok["input_ids"][j]
                attn = [1] * len(ids)
                ids_list[i] = ids
                attn_list[i] = attn
                self._ht_tok_cache[meta[i]] = (ids, attn)

        lengths = [len(x) for x in ids_list]
        max_len = max(lengths) if lengths else 0
        if max_len % 8 != 0:
            max_len = ((max_len + 7) // 8) * 8

        input_ids_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.model.device)
        attn_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.model.device)
        for i, ids in enumerate(ids_list):
            l = len(ids)
            if l == 0:
                continue
            input_ids_batch[i, -l:] = torch.tensor(ids, dtype=torch.long, device=self.model.device)
            attn_mask_batch[i, -l:] = 1

        with torch.no_grad():
            h_t_batch = self.word_embedding(input_ids_batch).to(torch.float16)  # [B, L, H] - 转换为float16避免bitsandbytes警告
        
        # === Use h_g directly as soft prompts (already projected by h_g_projector) ===
        # h_g is already [batch, hidden] from encode_graphs, no need for extra processing
        # soft_prompt = h_g  # [B, hidden] - TEMPORARILY DISABLED for testing

        # === Build LLM inputs with h_t sequence (h_t already contains full chat template) ===
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        pad_ids = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)
        pad_embeds = self.word_embedding(pad_ids).unsqueeze(0).to(torch.float16)

        for i in range(batch_size):
            # 取当前文本有效长度（由 attn mask 求和）
            L_i = int(attn_mask_batch[i].sum().item())
            h_t_i = h_t_batch[i, -L_i:, :]

            # 答案 ids/embeds
            label_tokens = self.tokenizer(labels[i], add_special_tokens=False)
            answer_input_ids = label_tokens.input_ids[:self.max_new_tokens] + eos_tokens.input_ids
            answer_embeds = self.word_embedding(torch.tensor(answer_input_ids, device=self.model.device)).to(torch.float16)

            # 根据开关决定是否使用图软提示 token
            if self.use_graph_prompt:
                soft_prompt_token = h_g[i].unsqueeze(0)  # [1, H]
                inputs_embeds = torch.cat([soft_prompt_token, h_t_i, answer_embeds], dim=0)
                prefix_length = 1 + L_i  # 包含soft prompt
            else:
                inputs_embeds = torch.cat([h_t_i, answer_embeds], dim=0)
                prefix_length = L_i  # 不包含soft prompt
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_ids = [IGNORE_INDEX] * inputs_embeds.shape[0]
            label_ids[prefix_length : prefix_length + len(answer_input_ids)] = answer_input_ids
            batch_label_input_ids.append(label_ids)

        # pad to max length in the batch
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_len + batch_label_input_ids[i]

        inputs_embeds   = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        # 仅训练投影器时，要求 inputs_embeds 支持反向传播以让梯度回传到 h_g_projector
        inputs_embeds.requires_grad_(True)
        attention_mask  = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # 步骤4: 计算统一的语言模型损失
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss
    def inference(self, *args, **kwargs):
        """
        推理模式（兼容两种调用）：
        1) 旧：model.inference({'subs': subs, 'qids': qids, 'question': questions})
        2) 新：model.inference(subs, qids, objs=None, mode='test', question_texts=questions)
        返回: {'pred': [str], 'question': [str], 'qids': qids}
        """
        # ---------- 解析入参：同时支持旧/新两种签名 ----------
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            # 旧：单个 batch 字典
            batch = args[0]
            subs = batch["subs"]
            qids = batch["qids"]
            questions = batch["question"]
        else:
            # 新：位置 + 关键字
            subs = kwargs.get('subs', None)
            qids = kwargs.get('qids', None)
            questions = kwargs.get('question_texts', None) or kwargs.get('questions', None)

            # 尝试从位置参数取 subs, qids
            if subs is None or qids is None:
                if len(args) >= 2:
                    subs, qids = args[:2]
                else:
                    raise ValueError("GraphLLM.inference 需要至少提供 subs 与 qids。")

            # 若未显式给出 questions，则用 loader.id2question 回填
            if questions is None:
                # qids 可能是 np.array / list，统一转 int
                try:
                    questions = [self.loader.id2question.get(int(q), "") for q in qids]
                except Exception:
                    questions = [""] * (len(qids) if hasattr(qids, '__len__') else 1)

            batch = {"subs": subs, "qids": qids, "question": questions}

        # ---------- 以下为你原来的实现（保持不变） ----------
        subs = batch["subs"]
        qids = batch["qids"]
        questions = batch["question"]

        # GNN编码（推理）- 使用缓存的 h_g_pooled 并过 projector
        self.graph_encoder.eval()
        h_g_pooled = self._get_hg_pooled_cached(subs, qids, mode='llm_inference')
        h_g = self.h_g_projector(h_g_pooled).to(torch.float16)  # 转换为float16

        # 批量构造输入文本并一次性嵌入
        texts = []
        ids_list, attn_list, meta = [], [], []
        pad_id = int(self.tokenizer.pad_token_id)
        batch_size = len(questions)
        for i, question in enumerate(questions):
            split, relative_qid, split_file = self._split_info(int(qids[i]))

            mc_prompt = self.load_multi_choice_prompt_from_file(relative_qid, split_file)

            cache_key = (split, int(relative_qid))
            if cache_key in self._ht_tok_cache:
                ids, attn = self._ht_tok_cache[cache_key]
                ids_list.append(ids)
                attn_list.append(attn)
                texts.append(None)
            else:
                full_text = self._build_chat_text(question, mc_prompt)
                texts.append(full_text)
                ids_list.append(None)
                attn_list.append(None)
            meta.append(cache_key)

        # 分词未缓存项（不填充），随后统一左填充
        uncached_idx = [i for i, ids in enumerate(ids_list) if ids is None]
        if len(uncached_idx) > 0:
            to_tok = [texts[i] for i in uncached_idx]
            tok = self.tokenizer(
                to_tok,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=self.max_txt_len,
            )
            for j, i in enumerate(uncached_idx):
                ids = tok["input_ids"][j]
                attn = [1] * len(ids)
                ids_list[i] = ids
                attn_list[i] = attn
                self._ht_tok_cache[meta[i]] = (ids, attn)

        lengths = [len(x) for x in ids_list]
        max_len = max(lengths) if lengths else 0
        if max_len % 8 != 0:
            max_len = ((max_len + 7) // 8) * 8

        input_ids_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.model.device)
        attn_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.model.device)
        for i, ids in enumerate(ids_list):
            l = len(ids)
            if l == 0:
                continue
            input_ids_batch[i, -l:] = torch.tensor(ids, dtype=torch.long, device=self.model.device)
            attn_mask_batch[i, -l:] = 1

        with torch.no_grad():
            h_t_batch = self.word_embedding(input_ids_batch).to(torch.float16)  # [B, L, H] - 转换为float16避免bitsandbytes警告

        # === Use h_g directly as soft prompts (already projected by h_g_projector) ===
        # h_g is already [batch, hidden] from encode_graphs, no need for extra processing
        # soft_prompt = h_g  # [B, hidden] - TEMPORARILY DISABLED for testing

        # === Build inputs: \hat{h_g} ⊕ h_t (h_t already contains chat template) ===
        batch_size = len(questions)
        batch_inputs_embeds = []
        batch_attention_mask = []

        pad_ids = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)
        pad_embeds = self.word_embedding(pad_ids).unsqueeze(0).to(torch.float16)

        for i in range(batch_size):
            L_i = int(attn_mask_batch[i].sum().item())
            h_t_i = h_t_batch[i, -L_i:, :]
            # 根据开关决定是否使用图软提示 token
            if self.use_graph_prompt:
                soft_prompt_token = h_g[i].unsqueeze(0)
                inputs_embeds = torch.cat([soft_prompt_token, h_t_i], dim=0)
            else:
                inputs_embeds = h_t_i
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            # Disable use_cache when gradient checkpointing is enabled to avoid warning
            use_cache = not self.model.config.use_cache if hasattr(self.model.config, 'gradient_checkpointing') and self.model.config.gradient_checkpointing else True
            
            # 生成配置：允许一定创造性来补充知识图谱空缺
            # 低温度采样：大部分时候选择最优答案，但允许LLM在证据不足时用内部知识补充
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=int(self.max_new_tokens),
                attention_mask=attention_mask,
                use_cache=use_cache,
                do_sample=True,       # 开启采样以允许创造性
                temperature=0.2,      # 低温度：保持高准确性，但允许适度变化
                top_p=0.9,          # 核采样：从累积95%概率的词中选择
                # 低温度+高top_p的组合：
                # - 当有明确答案时，仍会选择最优
                # - 当证据不足时，允许LLM探索其内部知识
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


        return {
            'pred': pred,
            'question': questions,
            'qids': qids,  # 用于评分. [3098]开始
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    
    def load_pretrained_gnn(self, pretrained_path):
        """加载预训练的GNN权重"""
        print(f"Loading pretrained GNN from {pretrained_path}")
        try:
            # safer loading if your torch supports it
            try:
                checkpoint = torch.load(pretrained_path, map_location=self.graph_encoder.device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(pretrained_path, map_location=self.graph_encoder.device)

            gnn_state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.graph_encoder.load_state_dict(gnn_state_dict, strict=False)
            print("Successfully loaded pretrained GNN weights!")
        except Exception as e:
            print(f"Error loading pretrained GNN: {e}")
            print("Continuing with randomly initialized GNN...")
    
    def freeze_gnn_parameters(self):
        """冻结GNN参数，只训练LLM和投影层"""
        print("Freezing GNN parameters...")
        frozen_params = 0
        total_gnn_params = 0
        for name, param in self.graph_encoder.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            total_gnn_params += param.numel()
        print(f"Frozen {frozen_params}/{total_gnn_params} GNN parameters")
        # 确保投影层参数可训练
        for name, param in self.h_g_projector.named_parameters():
            param.requires_grad = True
        # for name, param in self.soft_prompt_projector.named_parameters():
        #     param.requires_grad = True
