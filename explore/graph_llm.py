import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from models import Explore
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
        self.graph_encoder = Explore(args, loader)
        
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

    def encode_graphs(self, subs, qids, question_texts=None):
        """
        步骤1: GNN编码图结构 (对应graph_llm.py:108-116)
        """
        # 调用现有的GNN模型进行编码
        results = self.graph_encoder(subs, qids, mode='train', question_texts=question_texts)
        num_nodes, num_edges, scores_all, h_g, h_t, multi_choice_prompt = results
        
        return h_g, h_t, multi_choice_prompt

    def create_soft_prompt(self, h_g, h_t):
        """
        步骤2-3: 创建软提示 (对应graph_llm.py:133-144)
        soft_prompt = [h_g ; MEAN(h_t)]
        """
        batch_size = h_g.size(0)
        
        if h_t is not None:
            # 计算h_t的均值：[L, 4096] -> [1, 4096]
            h_t_mean = torch.mean(h_t, dim=0, keepdim=True)  # [1, 4096]
            h_t_mean = h_t_mean.expand(batch_size, -1)  # [batch_size, 4096]
        else:
            # 如果没有文本向量，使用零向量
            h_t_mean = torch.zeros(batch_size, 4096).to(h_g.device)
        
        # 拼接结构向量和文本向量: [h_g ; MEAN(h_t)]
        soft_prompt_input = torch.cat([h_g, h_t_mean], dim=-1)  # [batch_size, 8192]
        
        # 投影到LLM维度
        soft_prompt = self.soft_prompt_projector(soft_prompt_input)  # [batch_size, 4096]
        
        return soft_prompt

    def forward(self, batch):
        """
        统一前向传播和损失计算 (对应graph_llm.py:171)
        """
        subs = batch["subs"]
        qids = batch["qids"] 
        questions = batch["question"]
        labels = batch["label"]

        # 步骤1: GNN编码图结构
        h_g, h_t, multi_choice_prompt = self.encode_graphs(subs, qids, questions)

        # 步骤2: 创建软提示
        soft_prompt = self.create_soft_prompt(h_g, h_t)  # [batch_size, 4096]

        # 步骤3: 构建LLM输入
        batch_size = len(questions)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        # 编码特殊token
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        for i in range(batch_size):
            # 构建文本输入：多选提示 + 图结构 + 问题 + 答案
            if isinstance(multi_choice_prompt, list):
                mc_prompt = multi_choice_prompt[i] if i < len(multi_choice_prompt) else ""
            else:
                mc_prompt = multi_choice_prompt if multi_choice_prompt else ""
            
            # 使用graph_encoder的方法构建完整文本
            if hasattr(self.graph_encoder, 'textualize_subgraph'):
                desc = self.graph_encoder.textualize_subgraph(questions[i])
                if mc_prompt:
                    full_question = f"{mc_prompt}\n图结构:\n{desc}\nQuestion:\n{questions[i]}"
                else:
                    full_question = f"图结构:\n{desc}\nQuestion:\n{questions[i]}"
            else:
                full_question = questions[i]

            # 编码问题和答案
            question_tokens = self.tokenizer(full_question, add_special_tokens=False)
            label_tokens = self.tokenizer(labels[i], add_special_tokens=False)

            # 构建输入序列：[BOS] + soft_prompt + question + [/INST] + answer + [EOS]
            label_input_ids = label_tokens.input_ids[:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = question_tokens.input_ids[:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            
            # 获取文本embedding
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # 拼接：[BOS] + soft_prompt + text_embeds
            inputs_embeds = torch.cat([
                bos_embeds, 
                soft_prompt[i].unsqueeze(0),  # 软提示作为一个特殊token
                inputs_embeds
            ], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            
            # 设置标签：前面部分为IGNORE_INDEX，只对答案部分计算损失
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # 填充到相同长度
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
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

    def inference(self, batch):
        """推理模式"""
        subs = batch["subs"]
        qids = batch["qids"] 
        questions = batch["question"]

        # GNN编码
        h_g, h_t, multi_choice_prompt = self.encode_graphs(subs, qids, questions)

        # 创建软提示
        soft_prompt = self.create_soft_prompt(h_g, h_t)

        # 构建推理输入
        batch_size = len(questions)
        batch_inputs_embeds = []
        batch_attention_mask = []

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        for i in range(batch_size):
            # 构建完整问题
            if isinstance(multi_choice_prompt, list):
                mc_prompt = multi_choice_prompt[i] if i < len(multi_choice_prompt) else ""
            else:
                mc_prompt = multi_choice_prompt if multi_choice_prompt else ""
            
            if hasattr(self.graph_encoder, 'textualize_subgraph'):
                desc = self.graph_encoder.textualize_subgraph(questions[i])
                if mc_prompt:
                    full_question = f"{mc_prompt}\n图结构:\n{desc}\nQuestion:\n{questions[i]}"
                else:
                    full_question = f"图结构:\n{desc}\nQuestion:\n{questions[i]}"
            else:
                full_question = questions[i]

            question_tokens = self.tokenizer(full_question, add_special_tokens=False)
            input_ids = question_tokens.input_ids[:self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            inputs_embeds = torch.cat([
                bos_embeds, 
                soft_prompt[i].unsqueeze(0),
                inputs_embeds
            ], dim=0)
            
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # 填充
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'pred': pred,
            'question': questions,
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