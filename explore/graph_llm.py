import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models import Explore
from load_data import DataLoader
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

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

        print('Loading LLAMA')
        
        # 8bit量化配置
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        
        llm_kwargs = {
            "max_memory": {0: '40GiB',},
            "device_map": "auto",
            "revision": "main",
            "quantization_config": quant_config,  # 添加8bit量化
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=llm_kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **llm_kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
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

        self.model = model
        print('Finish loading LLAMA!')

        # 集成现有的GNN模型（Explore类）
        # 确保与GraphLLM使用相同的设备
        graph_device = self.model.device if hasattr(self.model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph_encoder = Explore(args, loader, device=graph_device)
        
        # 加载预训练的GNN权重
        if pretrained_gnn_path:
            self.load_pretrained_gnn(pretrained_gnn_path)
        
        # 冻结GNN参数
        if freeze_gnn:
            self.freeze_gnn_parameters()
        
        # h_g投影器：将GNN输出的h_g_pooled投影到LLM维度
        # h_g_pooled: [batch, hidden_dim] -> h_g: [batch, 4096]
        self.h_g_projector = nn.Sequential(
            nn.Linear(256, 2048),  # 假设hidden_dim=256
            nn.ReLU(),
            nn.Linear(2048, 4096)  # 投影到LLM维度
        ).to(self.model.device)
        
        # 软提示投影器：将[h_g ; MEAN(h_t)]投影到LLM维度
        # h_g: [batch, 4096], MEAN(h_t): [batch, 4096] -> soft_prompt: [batch, 4096]
        self.soft_prompt_projector = nn.Sequential(
            nn.Linear(4096 * 2, 2048),  # 拼接后的维度
            nn.ReLU(),
            nn.Linear(2048, 4096),  # 投影到LLM维度
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def load_multi_choice_prompt_from_file(self, qid, filepath='webqsp-train-path.txt'):
        """
        从文件中读取指定问题ID的多选提示（只读取第一条匹配的记录）
        
        Args:
            qid: 问题ID（相对索引）
            filepath: 包含多选提示的文件路径
        
        Returns:
            multi_choice_prompt: 该问题的多选提示字符串
        """
        # 添加缓存以避免重复读取
        if not hasattr(self, '_prompt_cache'):
            self._prompt_cache = {}
        
        # 检查缓存
        cache_key = (qid, filepath)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 每行格式：qid\t候选答案1|概率1|(路径1);...
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            line_qid = int(parts[0])
                            if line_qid == qid:
                                # 找到第一条匹配的记录，缓存并返回
                                result = '\t'.join(parts[1:])
                                self._prompt_cache[cache_key] = result
                                # print(result)
                                return result
                        except ValueError:
                            # 如果无法解析为整数，跳过这行
                            continue
        except Exception as e:
            print(f"Warning: Could not load multi-choice prompt from {filepath}: {e}")
        
        # 未找到时也缓存空字符串，避免重复查找
        self._prompt_cache[cache_key] = ""
        return ""
    
    def encode_graphs(self, subs, qids, question_texts=None):
        """
        步骤1: GNN编码图结构 (对应graph_llm.py:108-116)
        """
        # 调用现有的GNN模型进行编码
        results = self.graph_encoder(subs, qids, mode='llm_train', question_texts=question_texts)

        scores_all, h_g_pooled,  processed_subgraph = results

        # 通过h_g投影器将h_g_pooled投影到LLM维度
        h_g = self.h_g_projector(h_g_pooled)  # [batch_size, hidden_dim] -> [batch_size, 4096]

        return h_g,  processed_subgraph

    def generate_text_vector(self, question_text, multi_choice_prompt=None, processed_subgraph=None, max_seq_len=512):
        """
        步骤3: 文本向量生成
        多选提示 + 全局子图A文本化(CSV格式) + 问题 → LLM Token Embedding → h_t
        """
        # 使用传入的processed_subgraph，如果没有则使用textualize_subgraph方法
        if processed_subgraph is not None:
            desc = processed_subgraph
        elif hasattr(self.graph_encoder, 'textualize_subgraph'):
            desc = self.graph_encoder.textualize_subgraph(question_text)
        else:
            desc = ""
        
        # 构建完整的文本输入，按照流程.txt中的格式
        # 格式：多选提示 + 图结构(CSV) + Question: + 问题
        if multi_choice_prompt:
            full_text = f"多选提示：{multi_choice_prompt}\n图结构:{desc}\nQuestion:{question_text}"
        else:
            full_text = f"图结构:\n{desc}\nQuestion:\n{question_text}"
        
        # 获取Token Embedding
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding=False
        )
        
        with torch.no_grad():
            input_ids = inputs["input_ids"].to(self.model.device)
            h_t = self.word_embedding(input_ids).squeeze(0)  # [L, 4096]
        
        return h_t

    def create_soft_prompt(self, h_g):
        """
        Paper-style: the soft prompt is JUST the single graph token already
        projected to the LLM hidden size by self.h_g_projector.
        h_g: [batch, hidden]
        return: [batch, hidden]
        """
        return h_g

    def forward(self, batch):
        """
        统一前向传播和损失计算 (对应graph_llm.py:171)
        """
        subs = batch["subs"]
        qids = batch["qids"] 
        questions = batch["question"]
        labels = batch["label"]

        # 步骤1: GNN编码图结构（只获取h_g_pooled和子图）
        h_g, _, processed_subgraph = self.encode_graphs(subs, qids, questions)

        # 步骤2: 生成文本向量h_t
        h_t_list = []
        for i, question in enumerate(questions):
            # 从文件中读取多选提示
            # 根据数据集类型计算相对问题ID
            # 判断当前是训练集、验证集还是测试集
            current_qid = qids[i] if isinstance(qids, list) else qids[i].item()
            if current_qid < self.loader.n_train_qs:
                # 训练集
                relative_qid = current_qid
                filepath = 'webqsp-train-path.txt'
            elif current_qid < self.loader.n_train_qs + self.loader.n_valid_qs:
                # 验证集
                relative_qid = current_qid - self.loader.n_train_qs
                filepath = 'webqsp-dev-path.txt'
            else:
                # 测试集
                relative_qid = current_qid - self.loader.n_train_qs - self.loader.n_valid_qs
                filepath = 'webqsp-test-path.txt'
            
            multi_choice_prompt = self.load_multi_choice_prompt_from_file(relative_qid, filepath)
            
            h_t = self.generate_text_vector(
                question_text=question,
                multi_choice_prompt=multi_choice_prompt,  # 使用从文件读取的多选提示
                processed_subgraph=processed_subgraph  # 使用GNN生成的子图结构
            )
            h_t_list.append(h_t)
        
        # === Build soft prompts (graph-only) ===
        soft_prompts = []
        for i in range(len(questions)):
            h_g_single = h_g[i:i+1]  # [1, hidden]
            soft_prompt_single = self.create_soft_prompt(h_g_single)  # [1, hidden]
            soft_prompts.append(soft_prompt_single)
        soft_prompt = torch.cat(soft_prompts, dim=0)  # [B, hidden]

        # === Build LLM inputs with h_t sequence (NO extra tokenizer on the same text) ===
        batch_size = len(questions)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids.to(self.model.device)[0]
        pad_ids = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)

        bos_embeds = self.word_embedding(bos_ids)
        pad_embeds = self.word_embedding(pad_ids).unsqueeze(0)

        for i in range(batch_size):
            # h_t: [L, hidden] from generate_text_vector (textualized subgraph + prompts + question)
            h_t = h_t_list[i]

            # answer ids/embeds
            label_tokens      = self.tokenizer(labels[i], add_special_tokens=False)
            answer_input_ids  = label_tokens.input_ids[:self.max_new_tokens] + eos_tokens.input_ids
            answer_embeds     = self.word_embedding(torch.tensor(answer_input_ids, device=self.model.device))

            # single graph soft-token
            soft_prompt_token = soft_prompt[i].unsqueeze(0)  # [1, hidden]

            # final sequence: BOS ⊕ \hat{h_g} ⊕ h_t ⊕ answer (teacher forcing)
            inputs_embeds = torch.cat([bos_embeds, soft_prompt_token, h_t, answer_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

            # dynamic prefix length (BOS may be >1 tokens in some tokenizers)
            bos_len       = bos_embeds.shape[0]
            prefix_length = bos_len + 1 + h_t.shape[0]  # 1 = single graph token

            # labels: start with full IGNORE, only answer span carries targets
            label_ids = [IGNORE_INDEX] * inputs_embeds.shape[0]
            label_ids[prefix_length : prefix_length + len(answer_input_ids)] = answer_input_ids

            # # Optional one-time debug
            # if i == 0:
            #     print(f"[DBG] bos={bos_len}, h_t={h_t.shape[0]}, prefix={prefix_length}, total={inputs_embeds.shape[0]}, ans={len(answer_input_ids)}")

            batch_label_input_ids.append(label_ids)

        # pad to max length in the batch
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(batch_size):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_len + batch_label_input_ids[i]

        inputs_embeds   = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
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

        # GNN编码
        self.graph_encoder.eval()  # 推理时设置为eval模式
        results = self.graph_encoder(subs, qids, mode='llm_inference', question_texts=questions)
        h_g_pooled,  processed_subgraph = results
        h_g = self.h_g_projector(h_g_pooled)

        # 生成文本向量 h_t
        h_t_list = []
        for i, question in enumerate(questions):
            # 从文件中读取多选提示（以相对 qid 定位）
            try:
                q_i = int(qids[i])
            except Exception:
                q_i = qids[i]
            relative_qid = q_i - self.loader.n_train_qs - self.loader.n_valid_qs
            multi_choice_prompt = self.load_multi_choice_prompt_from_file(
                relative_qid, 'webqsp-path.txt'
            )

            h_t = self.generate_text_vector(
                question_text=question,
                multi_choice_prompt=multi_choice_prompt,  # 使用从文件读取的多选提示
                processed_subgraph=processed_subgraph  # 使用GNN生成的子图结构
            )
            h_t_list.append(h_t)

        # === Build soft prompts (graph-only) ===
        soft_prompts = []
        for i in range(len(questions)):
            h_g_single = h_g[i:i + 1]
            soft_prompt_single = self.create_soft_prompt(h_g_single)
            soft_prompts.append(soft_prompt_single)
        soft_prompt = torch.cat(soft_prompts, dim=0)

        # === Build inputs: BOS ⊕ \hat{h_g} ⊕ h_t ===
        batch_size = len(questions)
        batch_inputs_embeds = []
        batch_attention_mask = []

        bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids.to(self.model.device)[0]
        pad_ids = torch.tensor(self.tokenizer.pad_token_id, device=self.model.device)

        bos_embeds = self.word_embedding(bos_ids)
        pad_embeds = self.word_embedding(pad_ids).unsqueeze(0)

        for i in range(batch_size):
            h_t = h_t_list[i]  # [L, hidden]
            soft_prompt_token = soft_prompt[i].unsqueeze(0)
            inputs_embeds = torch.cat([bos_embeds, soft_prompt_token, h_t], dim=0)
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
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True,
                temperature=0,  # 匹配 example_chat_completion.py
                top_p=0.9,
                do_sample=False  # temperature=0 时不采样，确保确定性输出
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'question': questions,
            'qids': qids,  # 用于评分
        }



    def evaluate_with_scoring(self, dataset='webqsp'):
        """
        添加与example_chat_completion.py相同的评分机制
        """
        import json
        import numpy as np
        from pathlib import Path
        
        # 导入check.py的评分逻辑
        def candidate_path(root):
            all_entities = []
            all_scores = []
            all_paths = []
            all_ids = {}

            with open(root, 'r') as file:
                lines = file.readlines()

            i = 0
            for line in lines:
                line = line.strip()
                parts = line.split('\t')

                entities = []
                scores = []
                paths = []

                qid = int(parts[0])
                parts = parts[1:]
                for part in parts:
                    if part == '':
                        continue
                    try:
                        entity, score, path = part.split('|')
                    except:
                        print(part)
                        continue
                    entities.append(entity)
                    scores.append(float(score))
                    path = path.split(';')
                    split_path = []
                    for p in path:
                        if 'self_loop' not in p and p != '' and p not in split_path:
                            split_path.append(p)
                    split_path = ', '.join(split_path)
                    paths.append(split_path)

                all_entities.append(entities)
                all_scores.append(scores)
                all_paths.append(paths)
                all_ids[qid] = i
                i += 1

            return all_entities, all_scores, all_paths, all_ids
        
        def check_accuracy(dataset, predictions, prediction_ids):
            # 设置文件路径
            if dataset.startswith('MetaQA'):
                ta_file = f'../data/{dataset}/ntm/qa_test.txt'
                dataset = dataset.replace('/','_')
                path_file = f'../explore/{dataset}_path.txt'
            elif dataset == 'webqsp':
                ta_file = '../data/webqsp/Webqsp.txt'
                path_file = '../explore/webqsp_path.txt'
            elif dataset == 'CWQ':
                ta_file = '../data/CWQ/CWQ.txt'
                path_file = '../explore/CWQ_path.txt'

            # 读取真实答案
            all_ta = []
            with open(ta_file) as fta:
                for line in fta:
                    line = line.strip().split('\t')
                    try:
                        if dataset.startswith('WC'):
                            ta = line[2]
                            ta = ta.replace('/','|')[:-1]
                        else:
                            _, ta = line[0], line[1]
                        ta = ta.strip()
                    except:
                        ta = 'null'
                    all_ta.append(ta)

            # 读取候选路径
            all_candi, all_score, all_p, all_ids = candidate_path(path_file)

            # 计算准确率
            check = []
            check_abc = []
            n_true = 0
            n_null = 0
            
            for i, pred in enumerate(predictions):
                qid = prediction_ids[i] if i < len(prediction_ids) else i
                
                if qid >= len(all_ta):
                    continue
                    
                ta = all_ta[qid]
                if ta == 'null':
                    n_null += 1
                    check.append(0)
                    check_abc.append(0)
                    continue
                    
                ta_list = ta.split('|')
                flag = 0

                # 检查直接匹配
                for oneta in ta_list:
                    if oneta.lower() in pred.lower():
                        check.append(1)
                        n_true += 1
                        flag = 1
                        break
                if flag == 0:
                    check.append(0)

                # 检查ABC格式匹配
                flag = 0
                s = pred
                index_a = s.find('A. ')
                index_b = s.find('B. ')
                index_c = s.find('C. ')
                index_d = s.find('D. ')
                
                if qid not in all_ids:
                    check_abc.append(0)
                    continue
                    
                candi_idx = all_ids[qid]
                
                # 提取选择的答案
                if 0 <= index_a and (index_b == -1 or index_b > index_a) and (index_c == -1 or index_a < index_c) and (index_d == -1 or index_a < index_d):
                    extracted_answer = all_candi[candi_idx][0].lower()
                elif 0 <= index_b and (index_a == -1 or index_a > index_b) and (index_c == -1 or index_b < index_c) and (index_d == -1 or index_b < index_d):
                    extracted_answer = all_candi[candi_idx][1].lower()
                elif 0 <= index_c and (index_a == -1 or index_a > index_c) and (index_b == -1 or index_b > index_c) and (index_d == -1 or index_c < index_d):
                    extracted_answer = all_candi[candi_idx][2].lower()
                elif 0 <= index_d and (index_a == -1 or index_a > index_d) and (index_b == -1 or index_b > index_d) and (index_c == -1 or index_d < index_c):
                    extracted_answer = all_candi[candi_idx][3].lower()
                else:
                    extracted_answer = pred.lower()

                # 检查提取的答案是否匹配真实答案
                for oneta in ta_list:
                    if oneta.lower() in extracted_answer:
                        check_abc.append(1)
                        flag = 1
                        break
                if flag == 0:
                    check_abc.append(0)

            # 计算最终准确率
            total_valid = len(check) - n_null
            hit1abc = np.array(check_abc).sum() / total_valid if total_valid > 0 else 0
            hit = n_true / total_valid if total_valid > 0 else 0
            
            print(f'HIT@1 (ABC format): {hit1abc:.4f}')
            print(f'HIT (direct match): {hit:.4f}')
            
            return hit1abc, hit
        
        return check_accuracy

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
        for name, param in self.soft_prompt_projector.named_parameters():
            param.requires_grad = True