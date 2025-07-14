# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the llama 2 Community License Agreement.

from typing import List, Optional
import fire, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from utils import candidate_path
from check import check


def main(
        model_name: str = "meta-llama/Llama-2-13b-chat-hf",
        tokenizer_path: Optional[str] = None,
        temperature: float = 0,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        dataset: str = 'webqsp'
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    # 加载模型和tokenizer - 修复offload问题
    print(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path if tokenizer_path else model_name,
        padding_side="left"
    )

    # 确保有pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_state_dict=False,  # 防止参数被卸载到CPU
        low_cpu_mem_usage=True
    )

    # 如果添加了新的pad token，调整模型嵌入层大小
    if tokenizer.pad_token == "[PAD]":
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    print("Model loaded successfully.")

    # 增强的系统提示，包含创新点信息
    myprompt = {"role": "system",
                "content": "Given a question, and the reference answers with their correct probabilities, confidence scores, conflict detection results, and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this enhanced information and your own knowledge. The system provides: 1) Multi-granularity reasoning chains from dynamic path sampling, 2) Conflict-aware confidence calibration scores, 3) Graph-text alignment information. If the reference answers contain the correct answer, please output the label and content of the answer; If not, please answer the question based on your own knowledge."}

    myqs = {"role": "user", "content": ""}
    all_q = []
    # 数据加载部分保持不变
    if dataset.startswith('MetaQA'):  # MetaQA/1-hop
        f = open('../data/' + dataset + '/ntm/qa_test.txt')
        for line in f:
            line = line.strip().split('\t')
            q = line[0]
            q = q.replace('[', '')
            q = q.replace(']', '')
            q = q + "?"
            all_q.append(q)
        dataset = dataset.replace('/', '-')
        path_file = '../explore/' + dataset + '-path.txt'

    elif dataset == 'webqsp':
        mode = 'test'
        filepath = f'../data/webqsp/' + mode + '_simple.json'
        json_file = open(filepath, 'r')
        data = json_file.readlines()

        for line in data:
            entry = json.loads(line.strip())
            question = entry.get('question', '')
            question += '?'
            all_q.append(question)
        path_file = '../explore/webqsp-path.txt'

    elif dataset == 'CWQ':
        filepath = '../data/CWQ/test_simple.json'
        json_file = open(filepath, 'r')
        data = json_file.readlines()
        for line in data:
            entry = json.loads(line.strip())
            question = entry.get('question', '')
            all_q.append(question)
        path_file = '../explore/CWQ-path.txt'

    # 使用增强的candidate_path函数获取创新点信息
    path_results = candidate_path(path_file)
    if len(path_results) == 7:  # 新格式包含创新点信息
        all_candi, all_score, all_p, all_ids, all_confidence, all_conflict, all_reasoning = path_results
    else:  # 兼容旧格式
        all_candi, all_score, all_p, all_ids = path_results
        all_confidence = [[0.9] * len(candi) for candi in all_candi]
        all_conflict = [[0.1] * len(candi) for candi in all_candi]
        all_reasoning = [[""] * len(candi) for candi in all_candi]

    fout = open(dataset + '-ans.jsonl', 'w')

    # 批处理生成函数
    def generate_batch(dialogs_batch):
        # 构建LLaMA2聊天格式的提示
        prompts = []
        for dialog in dialogs_batch:
            # LLaMA2聊天格式: <s>[INST] <<SYS>>系统提示<</SYS>>用户消息 [/INST]
            formatted_prompt = "<s>"
            for idx, message in enumerate(dialog):
                if message["role"] == "system":
                    formatted_prompt += f"[INST] <<SYS>>\n{message['content']}\n<</SYS>>\n\n"
                elif message["role"] == "user":
                    # 最后一个用户消息后添加 [/INST] 标记
                    if idx == len(dialog) - 1:
                        formatted_prompt += f"{message['content']} [/INST]"
                    else:
                        formatted_prompt += f"{message['content']} "
            prompts.append(formatted_prompt)

        # Tokenize批处理
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            pad_to_multiple_of=8  # 更好的GPU效率
        )

        # 移到GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 设置生成参数
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_gen_len if max_gen_len else max_seq_len // 2,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # 生成文本
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)

        # 解码结果（跳过输入部分）
        decoded_outputs = []
        for i, output in enumerate(outputs):
            # 获取新生成的文本（跳过输入长度）
            input_len = inputs["input_ids"][i].size(0)
            generated_tokens = output[input_len:]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            decoded_outputs.append(decoded.strip())

        return decoded_outputs

    # 存储所有对话
    all_dialogs = []
    # 存储问题ID和原始问题内容
    dialog_info = []

    for qid in tqdm(range(len(all_q)), desc="Processing questions"):
        q = all_q[qid]
        myqs = {"role": "user", "content": ""}

        # 构建增强的用户消息内容，包含创新点信息
        if qid in all_ids:
            i = all_ids[qid]
            # 构建增强的ABC选项，包含创新点信息
            enhanced_options = []
            for j in range(min(3, len(all_candi[i]))):
                option_letter = chr(ord('A') + j)
                candidate = all_candi[i][j]
                probability = all_score[i][j]
                facts = all_p[i][j]
                confidence = all_confidence[i][j] if j < len(all_confidence[i]) else 0.9
                conflict = all_conflict[i][j] if j < len(all_conflict[i]) else 0.1
                reasoning = all_reasoning[i][j] if j < len(all_reasoning[i]) else ""

                # 创新点信息集成到选项中
                option_text = (f"{option_letter}. {candidate} "
                             f"(GNN probability: {probability:.3f}, "
                             f"confidence: {confidence:.3f}, "
                             f"conflict score: {conflict:.3f}) "
                             f"{{relevant facts: {facts}}} "
                             f"{{reasoning chain: {reasoning}}}")
                enhanced_options.append(option_text)

            hintABCpp = ' '.join(enhanced_options) + ' Answer: '
            myqs['content'] = 'Question: ' + q + ' Enhanced Reference: ' + hintABCpp
        else:
            myqs['content'] = 'Question: ' + q + ' Answer:'

        # 创建对话 [系统提示, 用户消息]
        dialog = [myprompt, myqs]
        all_dialogs.append(dialog)
        dialog_info.append({
            "qid": qid,
            "original_question": q
        })

        # 当积累足够批次或处理最后一个问题时进行生成
        if len(all_dialogs) >= max_batch_size or qid == len(all_q) - 1:
            # 生成当前批次
            results = generate_batch(all_dialogs)

            # 保存结果
            for j, (dialog, result) in enumerate(zip(all_dialogs, results)):
                info = dialog_info[j]
                data = {
                    'id': info["qid"],
                    'answer': result.replace('\n', ' '),
                    'question': dialog[1]['content'].replace('\n', ' ')
                }
                fout.write(json.dumps(data) + '\n')
                fout.flush()

            # 重置批次
            all_dialogs = []
            dialog_info = []

    fout.close()
    print(f"Results saved to {dataset}-ans.jsonl")
    check(dataset=dataset)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    fire.Fire(main)