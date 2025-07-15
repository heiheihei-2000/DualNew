# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
import fire, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
from utils import  candidate_path
from check import check


def main(
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
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
    # 加载tokenizer和model
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
        low_cpu_mem_usage=True
    )
    
    # 如果添加了新的pad token，调整模型嵌入层大小
    if tokenizer.pad_token == "[PAD]":
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()

    # myprompt = {"role": "system", "content": "Given a question, and the reference answers with their correct probabilities and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this information and your own knowledge. Please output the answer label and content directly. Do not output the correct probability, related facts and other words."}
    myprompt = {"role": "system",
                "content": "Given a question, and the reference answers with their correct probabilities and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this information and your own knowledge. If the reference answers contain the correct answer, please output the label and content of the answer; If not, please answer the question based on your own knowledge."}

    myqs = {"role": "user", "content": ""}
    dialogs = []
    all_q = []
    # dataset = 'webqsp'
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

    all_candi, all_score, all_p, all_ids = candidate_path(path_file)

    fout = open(dataset + '-ans.jsonl', '+a')
    for qid in tqdm(range(len(all_q))):

        q = all_q[qid]
        myqs = {"role": "user", "content": ""}

        if qid in all_ids:
            i = all_ids[qid]
            hintABpp = ' A. ' + all_candi[i][0] + ' (correct probability: ' + str(
                all_score[i][0]) + ')  {relevant facts: ' + all_p[i][0] + '}  B. ' + all_candi[i][
                           1] + ' (correct probability: ' + str(all_score[i][1]) + ')  {relevant facts: ' + all_p[i][
                           1] + '}   Answer: '
            hintABCpp = ' A. ' + all_candi[i][0] + ' (correct probability: ' + str(
                all_score[i][0]) + ')  {relevant facts: ' + all_p[i][0] + '}  B. ' + all_candi[i][
                            1] + ' (correct probability: ' + str(all_score[i][1]) + ')  {relevant facts: ' + all_p[i][
                            1] + '} C. ' + all_candi[i][2] + ' (correct probability: ' + str(
                all_score[i][2]) + ')  {relevant facts: ' + all_p[i][2] + '}  Answer: '

            myqs['content'] = 'Question: ' + q + ' Reference answer: ' + hintABCpp
        else:
            myqs['content'] = 'Question: ' + q + ' Answer:'

        i = qid
        if i % max_batch_size == 0:
            dialogs = []

        dialogs.append([myprompt, myqs])
        if i % max_batch_size == max_batch_size - 1 or i == len(all_q) - 1:
            # 使用transformers进行批量生成
            prompts = []
            for dialog in dialogs:
                # 构建LLaMA2聊天格式的提示
                formatted_prompt = "<s>"
                for idx, message in enumerate(dialog):
                    if message["role"] == "system":
                        formatted_prompt += f"[INST] <<SYS>>\n{message['content']}\n<</SYS>>\n\n"
                    elif message["role"] == "user":
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
            )

            # 移到GPU
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_gen_len if max_gen_len else max_seq_len // 2,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 解码结果
            j = 0
            for dialog, output in zip(dialogs, outputs):
                # 获取新生成的文本（跳过输入长度）
                input_len = inputs["input_ids"][j].size(0)
                generated_tokens = output[input_len:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                msg = dialog[-1]  # if have myprompt it is [1] else [0]
                msg['content'] = msg['content'].replace('\n', ' ')
                a = decoded.strip().replace('\n', ' ')
                data = {'id': i // max_batch_size * max_batch_size + j, 'answer': a, 'question': msg['content']}
                fout.write(json.dumps(data) + '\n')
                j += 1

    fout.close()
    check(dataset=dataset)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    fire.Fire(main)

