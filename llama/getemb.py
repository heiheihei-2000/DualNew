from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from typing import List, Optional
import fire
import os, json
import torch
import numpy as np


def main(
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        dataset: str = 'webqsp'
):
    device_id = 0
    torch.cuda.set_device(device_id)
    print(f"Using GPU {device_id} ({torch.cuda.get_device_name(device_id)})")

    # 加载模型和tokenizer - 修复offload问题
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_state_dict=False,  # 防止参数被卸载到CPU
        low_cpu_mem_usage=True
    )

    # 确保所有参数在GPU上
    if any(p.device.type == 'meta' for p in model.parameters()):
        model.to('cuda')

    model.eval()

    # 获取模型真实维度
    hidden_size = model.config.hidden_size
    print(f"Model hidden size: {hidden_size}")

    data = dataset
    dataName = data
    modes = ['test', 'train', 'dev', 'rel']

    for mode in modes:
        all_q = []

        # 原数据集加载代码
        if dataName == 'CWQ':
            # CWQ
            if mode != 'rel':
                filepath = f'../data/CWQ/' + mode + '_simple.json'
                json_file = open(filepath, 'r')
                data = json_file.readlines()
                for line in data:
                    # 解析JSON行数据
                    entry = json.loads(line.strip())
                    question = entry.get('question', '')
                    all_q.append(question)
            else:
                with open('../data/CWQ/relations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        rel = line.strip()
                        all_q.append(rel)

        elif dataName == 'webqsp':
            if mode != 'rel':
                filepath = f'../data/webqsp/' + mode + '_simple.json'
                json_file = open(filepath, 'r')
                data = json_file.readlines()
                for line in data:
                    # 解析JSON行数据
                    entry = json.loads(line.strip())
                    question = entry.get('question', '')
                    all_q.append(question)
            else:
                with open('../data/webqsp/relations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        rel = line.strip()
                        all_q.append(rel)

        elif dataName.startswith('MetaQA'):
            if mode != 'rel':
                f = open('../data/' + data + '/ntm/qa_' + mode + '.txt')
                for line in f:
                    line = line.strip().split('\t')
                    q = line[0]
                    q = q.replace('[', '')
                    q = q.replace(']', '')
                    q = q + "?"
                    all_q.append(q)
            else:
                with open('../data/MetaQA/kb.txt') as f:
                    for line in f:
                        _, r, _ = line.strip().split('|')
                        all_q.append(r)

        n = len(all_q)
        print(f"Processing {mode} mode with {n} items")
        all_emb = np.zeros((n, hidden_size))

        with torch.no_grad():
            for i in tqdm(range(len(all_q))):
                q = all_q[i]
                emb = get_emb(model, tokenizer, q, max_seq_len)
                all_emb[i, :] = emb.cpu().data.numpy()

        # 原保存代码
        if mode == 'dev':
            mode = 'valid'
        if dataName.startswith('MetaQA'):
            dataName = 'Meta-' + data[7] + 'm'
            if mode == 'rel':
                dataName = 'Meta'

        save_dir = '../embedding'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        save_path = f'{save_dir}/{dataset}-{mode}.npy'
        np.save(save_path, all_emb)
        print(f"Saved embeddings to {save_path}")


def get_emb(model, tokenizer, prompt: str, max_seq_len: int):
    """修复注意力掩码问题的嵌入获取函数"""
    # 1. 编码文本
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_seq_len,
        truncation=True,
        padding=False,  # 不填充
    ).to(model.device)

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # 2. 获取嵌入层输出
    with torch.no_grad():
        # 获取嵌入层输出
        embeddings = model.model.embed_tokens(input_ids)

        # 3. 创建正确的注意力掩码 - 修复索引错误的关键
        # Llama模型需要4维的因果掩码
        attention_mask = torch.ones(
            (1, 1, seq_len, seq_len),
            dtype=torch.bool,
            device=model.device
        )
        # 创建因果掩码（下三角矩阵）
        mask = torch.tril(attention_mask)

        # 4. 获取第一层Transformer输出 - 使用正确的掩码
        first_layer_output = model.model.layers[0](
            hidden_states=embeddings,
            attention_mask=mask,
            position_ids=torch.arange(0, seq_len, device=model.device).unsqueeze(0),
            use_cache=False
        )[0]

        # 5. 获取最后一层输出
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            output_hidden_states=True
        )
        last_layer_output = outputs.hidden_states[-1]

        # 6. 应用层归一化
        norm = model.model.norm
        first_layer_norm = norm(first_layer_output)
        last_layer_norm = norm(last_layer_output)

        # 7. 首尾层融合（50%权重）
        fused_output = 0.5 * first_layer_norm + 0.5 * last_layer_norm

        # 8. 平均池化
        emb = fused_output.mean(dim=1)

    return emb.squeeze(0)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    fire.Fire(main)