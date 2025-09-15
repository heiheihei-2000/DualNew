import os
import wandb
import gc
from tqdm import tqdm
import sys
import torch
import json
import pandas as pd
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR
# from graph_llm import load_multi_choice_prompt_from_file
from load_data import DataLoader as KGDataLoader
from graph_llm import GraphLLM
from models import Explore
from base_model import BaseModel
from check import check

def parse_args_graph_llm():
    """解析GNN-LLM训练参数"""
    parser = argparse.ArgumentParser(description="GNN-LLM Training Arguments")
    
    # Dataset and paths
    parser.add_argument('--dataset', type=str, default='webqsp', 
                       choices=['webqsp', 'CWQ', 'WebCWQ', 'MetaQA/1-hop', 'MetaQA/2-hop', 'MetaQA/3-hop'],
                       help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./results', 
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    
    # Model parameters
    parser.add_argument('--llm_model_path', type=str, 
                       default='meta-llama/Llama-2-13b-chat-hf',
                       help='Path to LLM model')
    # Accept both --llm_frozen and the common misspelling --llmfrezon
    parser.add_argument('--llm_frozen', '--llmfrezon', dest='llm_frozen', type=str, default='False',
                       choices=['True', 'False'],
                       help='Whether to freeze LLM parameters (alias: --llmfrezon)')
    parser.add_argument('--pretrained_gnn_path', type=str, default=None,
                       help='Path to pretrained GNN model')
    parser.add_argument('--freeze_gnn', action='store_true',
                       help='Whether to freeze GNN parameters')
    parser.add_argument('--use_graph_prompt', type=str, default='True',
                       choices=['True', 'False'],
                       help='Whether to use GNN soft prompt (h_g) in LLM input')
    
    # GNN parameters
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='GNN hidden dimension')
    parser.add_argument('--attn_dim', type=int, default=5,
                       help='GNN attention dimension')
    parser.add_argument('--n_layer', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--act', type=str, default='relu',
                       choices=['relu', 'tanh', 'idd'],
                       help='Activation function')
    parser.add_argument('--K', type=int, default=200,
                       help='Number of neighbors to sample')
    parser.add_argument('--sample', type=int, default=1,
                       help='Whether to use sampling')
    
    # LLM parameters
    parser.add_argument('--max_txt_len', type=int, default=256,
                       help='Maximum text sequence length')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum new tokens to generate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=20,
                       help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Number of warmup epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--project', type=str, default='GNN-LLM',
                       help='Wandb project name')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Whether to use wandb logging')
    parser.add_argument('--eval_only', action='store_true',
                       help='Skip training and only evaluate the best saved model')
    parser.add_argument('--use_8bit', action='store_true',
                       help='Use 8-bit quantization for LLM to reduce memory usage')
    parser.add_argument('--disable_tqdm', action='store_true',
                       help='Disable tqdm progress bars to avoid buffering issues')
    parser.add_argument('--load', action='store_true',
                       help='Load pretrained weights from checkpoint to continue training')
    parser.add_argument('--load_projector_only', action='store_true',
                       help='Only load projector parameters, skip LLM parameters')
    
    return parser.parse_args()


def seed_everything(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adjust_learning_rate(param_group, base_lr, step_ratio, args):
    """调整学习率（包含warmup）"""
    if step_ratio < args.warmup_epochs:
        lr = base_lr * step_ratio / args.warmup_epochs
    else:
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * (step_ratio - args.warmup_epochs) / (args.num_epochs - args.warmup_epochs)))
    param_group['lr'] = lr


def save_checkpoint(model, optimizer, epoch, args, is_best=False):
    """保存模型检查点"""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    
    dataset_name = args.dataset.replace('/', '-') if '/' in args.dataset else args.dataset
    if is_best:
        path = os.path.join(args.checkpoint_dir, f'best_model_{dataset_name}.pth')
    else:
        path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}_{dataset_name}.pth')
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

    # Convenience: when only training the projector, also dump its weights separately
    try:
        if getattr(args, 'llm_frozen', 'False') == 'True' and is_best:
            proj_path = os.path.join(args.checkpoint_dir, f'best_projector_{dataset_name}.pth')
            torch.save({'epoch': epoch, 'state_dict': model.h_g_projector.state_dict()}, proj_path)
            print(f"Projector-only weights saved to {proj_path}")
    except Exception as e:
        print(f"[WARN] Failed to save projector-only checkpoint: {e}")


def load_best_model(model, args):
    dataset_name = args.dataset.replace('/', '-') if '/' in args.dataset else args.dataset
    # path = os.path.join(args.checkpoint_dir, f'best_model_{dataset_name}.pth')
    path = os.path.join(args.checkpoint_dir, f'best_model_webqsp-old.pth')

    if os.path.exists(path):
        # 尽量使用 weights_only=True（新版本 PyTorch），不支持时回退
        # 修复：使用正确的设备映射
        device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)

        # 兼容两种保存格式：纯 state_dict 或 包含 'model_state_dict'
        state = checkpoint.get('model_state_dict', checkpoint)
        
        # 如果只加载投影器参数
        if hasattr(args, 'load_projector_only') and args.load_projector_only:
            print("Loading projector parameters only (skipping LLM parameters)...")
            # 只提取投影器相关的参数
            projector_state = {k: v for k, v in state.items() if 'h_g_projector' in k}
            if projector_state:
                missing, unexpected = model.load_state_dict(projector_state, strict=False)
                print(f"Loaded {len(projector_state)} projector parameters")
            else:
                print("Warning: No projector parameters found in checkpoint!")
                missing = []
                unexpected = []
        else:
            # 加载所有参数
            missing, unexpected = model.load_state_dict(state, strict=False)
        
        if missing or unexpected:
            print(f"[load_best_model] missing keys: {missing}, unexpected keys: {unexpected}")

        device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Best model loaded from {path}")
    else:
        print(f"No checkpoint found at {path}")
    return model



import numpy as np
import torch
import json
from tqdm import tqdm
from math import ceil
import os
import sys
from tqdm import tqdm
import math
def evaluate_model_batch(model, loader, args, data='test', eval_batch_size=None):




    """
    评估（按问题写出 <dataset>-ans.jsonl，随后调用 check.py 计算 HIT@1）：
    - tqdm 按“样本数”更新；
    - 仅收集 qid / question / 模型输出的文本 pred；
    - 写 ans.jsonl 时用“相对 qid”作为 id，且对同一相对 qid 去重（保留首次）。
    """


    model.eval()
    batch_size = eval_batch_size if eval_batch_size is not None else args.n_tbatch

    # 数据量
    if data == 'valid':
        n_data = loader.n_valid
    else:
        n_data = loader.n_test

    n_batch = n_data // batch_size + (n_data % batch_size > 0)

    # 输出路径（你后面 check.py 会用 <dataset>-ans.jsonl）
    dataset_name = args.dataset.replace('/', '-') if '/' in args.dataset else args.dataset
    check_output_path = f'{dataset_name}-ans.jsonl'

    # 累积
    all_preds = []
    all_qtexts = []
    all_qids = []

    # 进度条
    if getattr(args, 'disable_tqdm', False):
        iterator = range(n_batch)
    else:
        iterator = tqdm(range(n_batch), desc=f"Evaluating {data}", unit="batch")

    with torch.no_grad():
        for i in iterator:
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)

            subs, qids, objs = loader.get_batch(batch_idx, data=data)

            # 取问题文本
            questions = [loader.id2question.get(int(q), f"question_{int(q)}") for q in qids]

            # 调用你的 GraphLLM.inference（我们已经改成兼容两种签名）
            batch = {
                'subs': subs.tolist() if isinstance(subs, np.ndarray) else subs,
                'qids': qids.tolist() if isinstance(qids, np.ndarray) else qids,
                'question': questions,
            }
            output = model.inference(batch)

            preds = output['pred'] if isinstance(output['pred'], list) else [output['pred']]
            # 有些模型会回传 question，这里优先用你刚构造的 questions
            all_preds.extend(preds)
            all_qtexts.extend(questions)
            all_qids.extend(qids.tolist() if isinstance(qids, np.ndarray) else list(qids))

    # === 写 <dataset>-ans.jsonl：用 “相对 qid” 做 id，并对相对 qid 去重 ===
    # 相对 qid = 全局 qid - n_train_qs - n_valid_qs（对 test）
    seen = set()
    n_written = 0
    with open(check_output_path, 'w', encoding='utf-8') as f:
        for pred, qtext, qid in zip(all_preds, all_qtexts, all_qids):
            try:
                qid_int = int(qid)
            except Exception:
                qid_int = -1

            if data == 'test':
                rel_qid = qid_int - getattr(loader, 'n_train_qs', 0) - getattr(loader, 'n_valid_qs', 0)
            elif data == 'valid':
                rel_qid = qid_int - getattr(loader, 'n_train_qs', 0)
            else:
                rel_qid = qid_int

            if rel_qid < 0 or rel_qid in seen:
                continue
            seen.add(rel_qid)

            rec = {
                'id': rel_qid,
                'answer': (pred or "").replace('\n', ' '),
                'question': (qtext or "").replace('\n', ' '),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_written += 1

    print(f"\nSaved {n_written} predictions to {check_output_path}", flush=True)

    # === 用 check.py 计算 HIT@1（你原来的逻辑保留） ===
    print(f"\nCalculating Hit rate using check.py for {args.dataset}...", flush=True)
    from check import check
    check(dataset=args.dataset)

    # 从 check 的输出汇总里读回 HIT@1
    check_result_file = f'check-{dataset_name}-ans.jsonl'
    hit_rate = 0.0
    if os.path.exists(check_result_file):
        try:
            # check-xxx-ans.jsonl 的最后一行是 {'HIT@1': xxx, 'HIT': yyy}
            with open(check_result_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            if lines:
                summary = json.loads(lines[-1])
                hit_rate = float(summary.get('HIT@1', 0.0))
                print(f"[CHECK.PY] HIT@1: {hit_rate:.4f}")
        except Exception as e:
            print(f"[WARN] Failed to read {check_result_file}: {e}")

    return hit_rate


    # helper to obtain predicted topk entity ids from model output
    def preds_from_model_output(output, j, topk=10):
        """
        Tries to extract predicted entity id list for batch index j from the model output.
        Returns: list[int] (length up to topk)
        """
        # if output is dict, try various keys
        if isinstance(output, dict):
            # 1) direct topk indices
            if 'topk_indices' in output:
                arr = output['topk_indices']
                # arr expected shape (batch, k)
                try:
                    row = arr[j]
                    return [int(x) for x in np.asarray(row).tolist()]
                except Exception:
                    pass
            if 'preds_topk' in output:
                arr = output['preds_topk']
                try:
                    row = arr[j]
                    return [int(x) for x in np.asarray(row).tolist()]
                except Exception:
                    pass
            # 2) scores -> argsort
            if 'scores' in output:
                sc = output['scores']
                # sc shape (batch, n_ent)
                if torch.is_tensor(sc):
                    sc = sc.detach().cpu().numpy()
                sc = np.asarray(sc)
                row = sc[j]
                order = np.argsort(row)[::-1][:topk]
                return [int(x) for x in order.tolist()]
            # 3) 'pred' could be ids or textual; try to interpret
            if 'pred' in output:
                preds = output['pred']
                # preds may be list of lists, list of ints, or list of strings
                try:
                    candidate = preds[j]
                    # if candidate is list/ndarray of ints
                    if isinstance(candidate, (list, np.ndarray)):
                        return [int(x) for x in np.asarray(candidate).tolist()][:topk]
                    # if an int-like
                    try:
                        return [int(candidate)]
                    except Exception:
                        # if textual, cannot map automatically -> return empty (caller must adapt)
                        return []
                except Exception:
                    pass
        else:
            # if output is raw numpy array or torch.tensor of scores
            if torch.is_tensor(output):
                arr = output.detach().cpu().numpy()
                row = arr[j]
                order = np.argsort(row)[::-1][:topk]
                return [int(x) for x in order.tolist()]
            elif isinstance(output, np.ndarray):
                row = output[j]
                order = np.argsort(row)[::-1][:topk]
                return [int(x) for x in order.tolist()]

        # fallback empty
        return []

    # main loop
    for i in range(n_batch):
        start = i * batch_size
        end = min(n_data, (i + 1) * batch_size)
        batch_idx = np.arange(start, end)

        # get batch - loader.get_batch returns (subs, rels/qids, objs) in this repo
        subs, rels, objs = loader.get_batch(batch_idx, data=data)

        # treat rels as qids for consistency
        qids = rels

        # Move inputs to device if needed (model-specific). Many models expect numpy arrays; adjust if needed.
        # Try flexible inference call patterns
        output = None
        # The model API may differ; try common patterns:
        try:
            # If model has an 'inference' method that accepts (subs, qids, objs, mode)
            if hasattr(model, 'inference'):
                try:
                    output = model.inference(subs=subs, qids=qids, objs=objs, mode=data)
                except TypeError:
                    # try positional args
                    try:
                        output = model.inference(subs, qids, objs, data)
                    except TypeError:
                        output = model.inference(subs, qids, data)
            else:
                # try calling model(...) and interpret return
                try:
                    # some models return scores etc.
                    output = model(subs, qids, mode=data)
                except TypeError:
                    output = model(subs, qids)
        except Exception as e:
            # If inference failed, raise to notify user to adapt this call
            raise RuntimeError(f"Model inference call failed in evaluate_model_batch: {e}")

        # Normalize output: if model returns a tuple (scores, ...) convert to dict
        # handled later by preds_from_model_output

        # Extract predictions for each sample in batch
        # predicted_topk_ids: list of lists (len=batch_size, each list up to topk ints)
        predicted_topk_ids = []
        batch_k = 10
        for j in range(end - start):
            pred_ids = preds_from_model_output(output, j, topk=batch_k)
            predicted_topk_ids.append(pred_ids)

        # For optional writing: also try to get textual preds if available
        # If output contains textual preds under 'pred_text' or 'pred', capture them (best effort)
        pred_texts = []
        if isinstance(output, dict) and 'pred_text' in output:
            pred_texts = list(output['pred_text'])
        elif isinstance(output, dict) and 'pred' in output:
            # if pred elements are strings we can use them
            cand = output['pred']
            if isinstance(cand, (list, tuple)) and len(cand) >= (end - start):
                if all(isinstance(x, str) for x in cand):
                    pred_texts = cand[start - i*batch_size : start - i*batch_size + (end-start)]
                else:
                    pred_texts = [str(x) for x in cand[start - i*batch_size : start - i*batch_size + (end-start)]]
        # Otherwise, leave pred_texts empty and we'll write entity ids as string if asked.

        # Now evaluate sample by sample
        for j in range(end - start):
            qid = qids[j]
            objs_row = objs[j] if objs is not None else None
            gold_set = get_gold_set_for_qid(qid, objs_row)

            preds_ids = predicted_topk_ids[j]  # list of ints (may be empty)
            # top1 id
            pred1 = preds_ids[0] if len(preds_ids) > 0 else -1

            # update hit counts
            if len(gold_set) > 0:
                if pred1 in gold_set:
                    hit1 += 1
                if any(p in gold_set for p in preds_ids):
                    hit10 += 1
            else:
                # if gold set empty, treat as neither hit nor miss (you may want to count or skip)
                pass

            total += 1

            # collect for optional writing
            all_qids.append(int(qid) if qid is not None else -1)

            if pred_texts:
                all_preds.append(pred_texts[j] if j < len(pred_texts) else str(preds_ids))
            else:
                # write ids as joined string (for compatibility with check.py prefer textual answers,
                # but if we only have ids, write the first top1 entity name if loader has id2entity)
                if len(preds_ids) > 0:
                    try:
                        # get entity name
                        ent_name = loader.id2entity.get(int(preds_ids[0]), str(preds_ids[0]))
                        all_preds.append(ent_name)
                    except Exception:
                        all_preds.append(str(preds_ids))
                else:
                    all_preds.append("")

            # For question text, attempt to get from output or loader
            qtext = ""
            if isinstance(output, dict) and 'question' in output:
                try:
                    qtext = output['question'][j]
                except Exception:
                    qtext = ""
            else:
                # try loader mapping from qid -> text
                try:
                    if data == 'test' and hasattr(loader, 'id2question'):
                        # qid may be relative or global; try direct
                        qtext = loader.id2question.get(int(qid), "")
                except Exception:
                    qtext = ""
            all_qtexts.append(qtext)

    # End batches loop

    # Compute rates
    hit1_rate = hit1 / total if total > 0 else 0.0
    hit10_rate = hit10 / total if total > 0 else 0.0

    metrics = {'hit1': hit1, 'hit10': hit10, 'total': total, 'hit1_rate': hit1_rate, 'hit10_rate': hit10_rate}
    print(f"[EVAL] total={total}  H@1={hit1_rate:.4f}  H@10={hit10_rate:.4f}")

    # Optionally write ans jsonl aligned by relative qid (dedup keep first seen)
    if write_ans_file:
        seen = set()
        with open(ans_out_path, 'w', encoding='utf-8') as wf:
            for pred_text, qtext, qid in zip(all_preds, all_qtexts, all_qids):
                # compute relative qid for test/valid to align with check.py expectation
                try:
                    qid_int = int(qid)
                except Exception:
                    qid_int = -1
                if data == 'test':
                    rel_qid = qid_int - getattr(loader, 'n_train_qs', 0) - getattr(loader, 'n_valid_qs', 0)
                elif data == 'valid':
                    rel_qid = qid_int - getattr(loader, 'n_train_qs', 0)
                else:
                    rel_qid = qid_int
                if rel_qid in seen or rel_qid < 0:
                    continue
                seen.add(rel_qid)
                record = {'id': rel_qid, 'answer': pred_text.replace('\n', ' '), 'question': qtext.replace('\n', ' ')}
                print(record)
                wf.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Wrote predictions to {ans_out_path} (deduped by relative qid)")

    return metrics





def main(args):
    """主训练函数"""
    # GraphLLM.load_multi_choice_prompt_from_file(self)
    # 设置设备（与train.py相同）
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    
    # 设置随机种子（与train.py相同）
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # # 初始化wandb
    # if args.use_wandb:
    #     wandb.init(
    #         project=args.project,
    #         name=f"{args.dataset}_GraphLLM_seed{args.seed}",
    #         config=vars(args)
    #     )
    
    print("Arguments:", flush=True)
    for key, value in vars(args).items():
        print(f"  {key}: {value}", flush=True)
    sys.stdout.flush()
    
    # 创建结果目录（与train.py相同）
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 创建Options对象（与train.py相同）
    class Options(object):
        pass
    
    opts = Options
    opts.perf_file = os.path.join(results_dir, args.dataset.replace('/', '-') + '_perf.txt')
    
    # 根据数据集设置参数（与train.py完全相同）
    dataset = args.dataset
    if dataset == 'MetaQA/1-hop':
        opts.lr = 0.00005
        opts.decay_rate = 0.996
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 1
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 40
        loaders = [KGDataLoader(args.dataset)]
    elif dataset == 'MetaQA/2-hop':
        opts.lr = 0.00004
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 2
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 60
        loaders = [KGDataLoader(args.dataset)]
    elif dataset == 'MetaQA/3-hop':
        opts.lr = 0.00002
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 100
        loaders = [KGDataLoader(args.dataset)]
    elif dataset == 'webqsp':
        opts.lr = 0.00001
        opts.decay_rate = 0.9991
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 1# Start with batch size 1 to debug
        opts.n_tbatch = 2

        opts.K = 200
        loaders = [KGDataLoader(args.dataset)]
    elif dataset == 'CWQ':
        opts.lr = 0.00001
        opts.decay_rate = 0.993
        opts.lamb = 0.0001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        loaders = [KGDataLoader(args.dataset)]
    elif dataset == 'WebCWQ':  # Combined webqsp and CWQ
        opts.lr = 0.0001
        opts.decay_rate = 0.9968
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.2
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        # Load both webqsp_nsm and CWQ datasets
        loaders = [KGDataLoader('webqsp_nsm'), KGDataLoader('CWQ')]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    opts.sample = 1
    opts.n_ent = loaders[0].n_ent
    opts.n_rel = loaders[0].n_rel
    
    # 打印配置信息（与train.py相同）
    config_str = '%d, %d, %.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (
        opts.sample, opts.K, opts.lr, opts.decay_rate, opts.lamb, 
        opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch,
        opts.dropout, opts.act
    )
    print(config_str.strip())
    
    # 保存配置信息
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)
    
    # 加载数据（与train.py完全相同）
    print(f"Loading dataset: {args.dataset}")
    loader = loaders[0]  # 使用与train.py相同的loader
    
    # 获取数据集大小（与base_model.py相同）
    n_train = loader.n_train
    n_valid = loader.n_valid
    n_test = loader.n_test
    
    print(f"Train samples: {n_train}")
    print(f"Valid samples: {n_valid}")
    print(f"Test samples: {n_test}")
    print(f"\nGradient Accumulation Steps: {args.grad_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_steps}")
    
    # 构建模型
    print("Building GraphLLM model...")
    # 将opts参数传递给args
    args.hidden_dim = opts.hidden_dim
    args.attn_dim = opts.attn_dim
    args.n_layer = opts.n_layer
    args.dropout = opts.dropout
    args.act = opts.act
    args.K = opts.K
    args.sample = opts.sample
    args.n_ent = opts.n_ent
    args.n_rel = opts.n_rel
    args.n_batch = opts.n_batch
    args.n_tbatch = opts.n_tbatch
    
    # 设置批次大小（与train.py保持一致）
    batch_size = opts.n_batch
    eval_batch_size = opts.n_tbatch
    
    # 处理 WebCWQ 数据集的多个 GNN 权重路径
    if dataset == 'WebCWQ':
        # WebCWQ 需要为不同数据集准备不同的 GNN 权重
        pretrained_gnn_path = {
            'webqsp': 'webqsp_best_saved_model.pt',  # WebQSP 对应的 GNN 权重
            'cwq': 'CWQ_best_saved_model.pt'  # CWQ 对应的 GNN 权重（键名统一小写）
        }
        print("Using multiple GNN weights for WebCWQ:")
        print(f"  webqsp: {pretrained_gnn_path['webqsp']}")
        print(f"  cwq:    {pretrained_gnn_path['cwq']}")
    else:
        # 单个数据集使用单个 GNN 权重
        pretrained_gnn_path = args.pretrained_gnn_path
    
    model = GraphLLM(
        args=args, 
        loader=loader,
        pretrained_gnn_path=pretrained_gnn_path,
        freeze_gnn=args.freeze_gnn
    )
    # model.load_multi_choice_prompt_from_file()
    # 打印可训练参数
    trainable_params, all_params = model.print_trainable_params()
    print(f"Trainable params: {trainable_params} || All params: {all_params} || "
          f"Trainable%: {100 * trainable_params / all_params:.2f}%")
    print(f"\nTraining configuration:")
    print(f"  Micro batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {args.grad_steps}")
    print(f"  Effective batch size: {batch_size * args.grad_steps}")
    
    # 设置优化器（与train.py保持一致）
    # 当 LLM 冻结时，仅训练 h_g_projector（满足“只训练 h_g_projector”的需求）
    if str(args.llm_frozen) == 'True':
        params = list(model.h_g_projector.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=opts.lr, weight_decay=opts.lamb)
    
    # 设置学习率调度器（与train.py相同）

    scheduler = ExponentialLR(optimizer, opts.decay_rate)
    
    # 如果需要加载预训练权重
    start_epoch = 0
    if args.load:
        # Use args.checkpoint_dir to locate the best checkpoint for resuming
        dataset_name = args.dataset.replace('/', '-') if '/' in args.dataset else args.dataset
        # checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_{dataset_name}.pth')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_webqsp.pth')

        if os.path.exists(checkpoint_path):
            print(f"\nLoading pretrained weights from {checkpoint_path}...")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.gpu}', weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.gpu}')
            
            # 加载模型状态
            model_state = checkpoint.get('model_state_dict', checkpoint)
            
            # 如果只加载投影器参数
            if args.load_projector_only:
                print("Loading projector parameters only (skipping LLM parameters)...")
                # 只提取投影器相关的参数
                projector_state = {k: v for k, v in model_state.items() if 'h_g_projector' in k}
                if projector_state:
                    missing_keys, unexpected_keys = model.load_state_dict(projector_state, strict=False)
                    print(f"Loaded {len(projector_state)} projector parameters")
                else:
                    print("Warning: No projector parameters found in checkpoint!")
                    missing_keys = []
                    unexpected_keys = []
            else:
                # 加载所有参数
                missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading model: {unexpected_keys}")
            
            # 加载优化器状态（如果存在）
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state")
            
            # 获取开始的epoch（如果存在）
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")
            
            print(f"Successfully loaded pretrained weights from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            print("Starting training from scratch...")
    
    # 如果只是评估模式，跳过训练
    if args.eval_only:
        print("\nEval-only mode: Skipping training...")
        # 加载最佳模型
        print("Loading best model for evaluation...")
        # model = load_best_model(model, args)
        
        # 清理GPU内存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 加载模型
        model = load_best_model(model, args)






        # 最终评估（使用与train.py相同的评估方式）
        print("Running evaluation on test set...", flush=True)
        sys.stdout.flush()
        try:
            test_hit_rate = evaluate_model_batch(model, loader, args, data='test', eval_batch_size=eval_batch_size)
            print(f"\n{'='*50}")
            print(f"FINAL TEST RESULTS")
            print(f"{'='*50}")
            print(f"Test Hit@1 Rate: {test_hit_rate:.4f}%")
            print(f"{'='*50}\n")
            
            # 保存结果到文件
            result_file = os.path.join(args.output_dir, f'{args.dataset}_eval_results.txt')
            with open(result_file, 'w') as f:
                f.write(f"Test Hit@1 Rate: {test_hit_rate:.4f}%\n")
            print(f"Results saved to: {result_file}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理内存
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            gc.collect()
        return
    
    # 训练循环（与train.py的train_batch方法保持一致）
    best_train_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(start_epoch, args.num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        total_batches = 0
        
        # 对于 WebCWQ，需要训练多个数据集
        if dataset == 'WebCWQ' and len(loaders) > 1:
            print(f'\nEpoch {epoch+1} - Training on multiple datasets (WebCWQ)')
            
            # 遍历每个 loader
            for loader_idx, current_loader in enumerate(loaders):
                # 更新模型中的 loader（这样 _split_info 能正确工作）
                model.loader = current_loader
                
                # 打乱训练数据
                current_loader.shuffle_train()
                
                # 计算批次数
                n_batch = current_loader.n_train // batch_size + (current_loader.n_train % batch_size > 0)
                
                # MetaQA 特殊处理
                if 'MetaQA/2-hop' in current_loader.task_dir or 'MetaQA/3-hop' in current_loader.task_dir:
                    n_batch = n_batch // 10
                
                dataset_name = 'webqsp' if 'webqsp' in current_loader.task_dir.lower() else 'CWQ'
                print(f'  Dataset {loader_idx+1}/{len(loaders)} ({dataset_name}): {n_batch} batches')
                
                # 使用条件tqdm
                iterator = range(n_batch)
                if not hasattr(args, 'disable_tqdm') or not args.disable_tqdm:
                    iterator = tqdm(iterator, desc=f"Training {dataset_name}")
                
                # Gradient accumulation settings
                accumulation_steps = args.grad_steps
                if loader_idx == 0:  # 只在第一个 loader 时清零梯度
                    optimizer.zero_grad()
                accumulated_loss = 0.0
                
                for i in iterator:
                    start = i * batch_size
                    end = min(current_loader.n_train, (i + 1) * batch_size)
                    batch_idx = np.arange(start, end)
                    
                    # 使用get_batch获取数据
                    subs, qids, objs = current_loader.get_batch(batch_idx)
                    
                    # 转换为GraphLLM期望的格式
                    questions = []
                    labels = []
                    for j, q_id in enumerate(qids):
                        question_text = current_loader.id2question.get(q_id, f"question_{q_id}")
                        questions.append(question_text)
                        
                        # 处理答案
                        answer_entity = objs[j]
                        answer_text = current_loader.id2entity.get(answer_entity, f"entity_{answer_entity}")
                        labels.append(answer_text)
                    
                    batch = {
                        'subs': subs.tolist() if isinstance(subs, np.ndarray) else subs,
                        'qids': qids.tolist() if isinstance(qids, np.ndarray) else qids,
                        'question': questions,
                        'label': labels
                    }
                    
                    # Forward pass and scale loss by accumulation steps
                    loss = model(batch)
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    
                    # Perform optimizer step every accumulation_steps iterations
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batch:
                        # 梯度裁剪
                        clip_grad_norm_(params, max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        # Add accumulated loss to epoch loss
                        epoch_loss += accumulated_loss * accumulation_steps
                        accumulated_loss = 0.0
                    
                    # Update progress bar
                    if not hasattr(args, 'disable_tqdm') or not args.disable_tqdm:
                        iterator.set_postfix({'loss': accumulated_loss * accumulation_steps})
                    
                    # 定期清理内存
                    if (i + 1) % 10 == 0:
                        torch.cuda.empty_cache()
                
                total_batches += n_batch
        else:
            # 单个数据集的原始逻辑
            loader = loaders[0]
            
            # 打乱训练数据
            loader.shuffle_train()
            
            # 计算批次数
            n_batch = loader.n_train // batch_size + (loader.n_train % batch_size > 0)
            print(f'Epoch {epoch+1}, n_batch: {n_batch}')
            
            # MetaQA 特殊处理
            if 'MetaQA/2-hop' in loader.task_dir or 'MetaQA/3-hop' in loader.task_dir:
                n_batch = n_batch // 10
            
            # 使用条件tqdm
            iterator = range(n_batch)
            if not hasattr(args, 'disable_tqdm') or not args.disable_tqdm:
                iterator = tqdm(iterator, desc="Training")
            
            # Gradient accumulation settings
            accumulation_steps = args.grad_steps
            optimizer.zero_grad()
            accumulated_loss = 0.0
            
            for i in iterator:
                start = i * batch_size
                end = min(loader.n_train, (i + 1) * batch_size)
                batch_idx = np.arange(start, end)
                
                # 使用get_batch获取数据（与base_model.py相同）
                subs, qids, objs = loader.get_batch(batch_idx)
                
                # 转换为GraphLLM期望的格式
                questions = []
                labels = []
                for j, q_id in enumerate(qids):
                    question_text = loader.id2question.get(q_id, f"question_{q_id}")
                    questions.append(question_text)
                    
                    # 处理答案 - train_data中每个样本已经是单个答案
                    answer_entity = objs[j]
                    # objs[j]应该是单个答案实体ID（因为train_data在read_web_qa中已经拆分）
                    answer_text = loader.id2entity.get(answer_entity, f"entity_{answer_entity}")
                    labels.append(answer_text)
                
                batch = {
                    'subs': subs.tolist() if isinstance(subs, np.ndarray) else subs,
                    'qids': qids.tolist() if isinstance(qids, np.ndarray) else qids,
                    'question': questions,
                    'label': labels
                }
                
                # Forward pass and scale loss by accumulation steps
                loss = model(batch)
                loss = loss / accumulation_steps
                loss.backward()
                
                accumulated_loss += loss.item()
                
                # Perform optimizer step every accumulation_steps iterations
                if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batch:
                    # 梯度裁剪
                    clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Add accumulated loss to epoch loss (multiply back by accumulation_steps)
                    epoch_loss += accumulated_loss * accumulation_steps
                    accumulated_loss = 0.0
                
                # Update progress bar with current accumulated loss
                if not hasattr(args, 'disable_tqdm') or not args.disable_tqdm:
                    iterator.set_postfix({'loss': accumulated_loss * accumulation_steps})
                
                # 定期清理内存
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            total_batches = n_batch
        
        # 学习率衰减（与train.py相同）
        scheduler.step()
        
        # 计算平均损失时使用总批次数
        avg_train_loss = epoch_loss / total_batches if total_batches > 0 else epoch_loss
        print(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # 不再进行validation评估，直接基于训练损失保存模型
        
        # 保存最佳模型（基于训练损失）
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, args, is_best=True)
            print(f"New best model saved! Train Loss: {best_train_loss:.4f}")
        
        # 早停检查
        if epoch - best_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, args, is_best=False)
        
        # 每个epoch结束后清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    # 训练完成，清理资源
    print(f"Training completed! Best Train Loss: {best_train_loss:.4f} at epoch {best_epoch+1}")
    
    # if args.use_wandb:
    #     wandb.log({'Best Train Loss': best_train_loss})
    #     wandb.finish()
    
    # 清理内存
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()


if __name__ == "__main__":
    args = parse_args_graph_llm()
    main(args)
