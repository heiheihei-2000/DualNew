import os

import numpy as np
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import argparse

from graph_llm import GraphLLM
from load_data import DataLoader as KGDataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='webqsp', choices=['webqsp', 'MetaQA', 'CWQ'])
    parser.add_argument('--data_path', type=str, default='../data/')
    
    # Model arguments
    parser.add_argument('--llm_model_path', type=str, default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument('--llm_frozen', type=str, default='True', choices=['True', 'False'])
    parser.add_argument('--max_txt_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    
    # GNN arguments
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--attn_dim', type=int, default=200)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--act', type=str, default='relu')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--project', type=str, default='DualR-Unified')
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


def collate_fn(batch):
    """数据整理函数 - 适配现有DataLoader接口"""
    if isinstance(batch[0], dict):
        # 新格式数据
        subs = [item['sub'] for item in batch]
        qids = [item['qid'] for item in batch]
        questions = [item['question'] for item in batch]
        labels = [item['label'] for item in batch]
    else:
        # 现有DataLoader格式：(subs, qids, objs)
        subs, qids, objs = batch
        questions = qids  # 暂时使用qids作为问题，后续会被替换
        labels = objs     # 暂时使用objs作为标签，后续会被替换
    
    return {
        'subs': subs,
        'qids': qids,
        'question': questions,
        'label': labels
    }


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer_group, base_lr, step, epoch, args):
    """学习率调度"""
    # 简单的线性warmup + cosine衰减
    warmup_steps = 100
    total_steps = args.num_epochs * 1000  # 估算
    
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    optimizer_group['lr'] = lr


def _save_checkpoint(model, optimizer, epoch, args, is_best=False):
    """保存checkpoint"""
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args
    }
    
    if is_best:
        torch.save(checkpoint, f'{args.output_dir}/{args.dataset}/best_model.pt')
    else:
        torch.save(checkpoint, f'{args.output_dir}/{args.dataset}/checkpoint_epoch_{epoch}.pt')


def _reload_best_model(model, args):
    """重新加载最佳模型"""
    checkpoint_path = f'{args.output_dir}/{args.dataset}/best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {checkpoint_path}")
    return model


def create_sample_data(loader, split='train'):
    """创建样本数据用于调试"""
    samples = []
    n_samples = 100 if split == 'train' else 20
    
    for i in range(min(n_samples, getattr(loader, f'n_{split}'))):
        # 这里需要根据实际的数据加载器接口调整
        sample = {
            'sub': [i],  # 示例subject实体
            'qid': i,    # 问题ID
            'question': f"Sample question {i}",  # 示例问题
            'label': f"Sample answer {i}"        # 示例答案
        }
        samples.append(sample)
    
    return samples


def main(args):
    # 设置随机种子
    seed_everything(args.seed)
    
    # 初始化wandb
    wandb.init(project=args.project,
               name=f"{args.dataset}_unified_seed{args.seed}",
               config=args)
    
    print("Arguments:", args)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    
    # 加载数据
    print("Loading data...")
    try:
        # 使用现有的数据加载器
        kg_loader = KGDataLoader(args.dataset)
        args.n_rel = kg_loader.n_rel
        args.n_ent = kg_loader.n_ent
        
        # 直接使用现有DataLoader的批处理接口
        def get_batch_wrapper(batch_size, data_type='train'):
            """包装器函数，使用现有DataLoader的get_batch方法"""
            if data_type == 'train':
                n_samples = kg_loader.n_train
            elif data_type == 'valid':
                n_samples = kg_loader.n_valid
            else:  # test
                n_samples = kg_loader.n_test
            
            batch_indices = list(range(0, n_samples, batch_size))
            for start_idx in batch_indices:
                end_idx = min(start_idx + batch_size, n_samples)
                batch_idx = list(range(start_idx, end_idx))
                subs, qids, objs = kg_loader.get_batch(batch_idx, data_type)
                
                # 转换为需要的格式，添加真实的问题文本
                batch_data = []
                for i in range(len(batch_idx)):
                    # 从DataLoader获取真实问题文本
                    question_text = kg_loader.id2question.get(qids[i], f"Question {qids[i]}")
                    
                    # 简单的答案文本生成（实际应用中需要根据objs生成）
                    if isinstance(objs[i], (list, np.ndarray)) and len(objs[i]) > 0:
                        # 如果objs是one-hot向量，找到对应的实体名称
                        if hasattr(kg_loader, 'id2entity'):
                            try:
                                answer_indices = np.where(np.array(objs[i]) > 0)[0]
                                if len(answer_indices) > 0:
                                    answer_text = kg_loader.id2entity.get(answer_indices[0], f"Entity {answer_indices[0]}")
                                else:
                                    answer_text = f"Answer {i}"
                            except:
                                answer_text = f"Answer {i}"
                        else:
                            answer_text = f"Answer {i}"
                    else:
                        answer_text = f"Answer {i}"
                    
                    batch_data.append({
                        'sub': subs[i] if isinstance(subs[i], list) else [subs[i]],
                        'qid': qids[i],
                        'question': question_text,
                        'label': answer_text
                    })
                yield batch_data
        
        # 创建数据生成器
        train_batches = list(get_batch_wrapper(args.batch_size, 'train'))
        val_batches = list(get_batch_wrapper(args.eval_batch_size, 'valid'))
        test_batches = list(get_batch_wrapper(args.eval_batch_size, 'test'))
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data for testing...")
        
        # 使用虚拟数据进行测试
        args.n_rel = 100
        args.n_ent = 1000
        train_samples = [{'sub': [i], 'qid': i, 'question': f"Q{i}", 'label': f"A{i}"} for i in range(100)]
        val_samples = [{'sub': [i], 'qid': i, 'question': f"Q{i}", 'label': f"A{i}"} for i in range(20)]
        test_samples = [{'sub': [i], 'qid': i, 'question': f"Q{i}", 'label': f"A{i}"} for i in range(20)]
        kg_loader = None
        
        # 创建数据加载器
        train_loader = DataLoader(train_samples, batch_size=args.batch_size, 
                                 shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_samples, batch_size=args.eval_batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_samples, batch_size=args.eval_batch_size, 
                                shuffle=False, collate_fn=collate_fn)
    
    # 创建统一模型
    print("Creating unified GraphLLM model...")
    model = GraphLLM(args, kg_loader)
    
    # 设置优化器
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}],
        betas=(0.9, 0.95)
    )
    
    trainable_params, all_param = model.print_trainable_params()
    print(f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}")
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    
    if kg_loader:
        # 使用真实数据训练
        total_steps = 0
        for epoch in range(args.num_epochs):
            # 训练
            model.train()
            epoch_loss = 0.0
            accum_loss = 0.0
            step_count = 0
            
            for batch_data in train_batches:
                optimizer.zero_grad()
                
                try:
                    # 应用collate_fn处理批数据
                    batch = collate_fn(batch_data)
                    # 统一损失计算：loss = model(batch)
                    loss = model(batch)
                    loss.backward()
                    
                    clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    
                    if (step_count + 1) % args.grad_steps == 0:
                        adjust_learning_rate(optimizer.param_groups[0], args.lr, 
                                           total_steps, epoch, args)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    accum_loss += loss.item()
                    
                    if (step_count + 1) % args.grad_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        wandb.log({'Lr': lr})
                        wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                        accum_loss = 0.0
                    
                    step_count += 1
                    total_steps += 1
                    
                except Exception as e:
                    print(f"Training error at step {step_count}: {e}")
                    continue
            
            avg_train_loss = epoch_loss / len(train_batches) if len(train_batches) > 0 else 0
            print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss: {avg_train_loss}")
            wandb.log({'Train Loss (Epoch Mean)': avg_train_loss})
            
            # 验证
            model.eval()
            val_loss = 0.0
            val_step_count = 0
            with torch.no_grad():
                for batch_data in val_batches:
                    try:
                        batch = collate_fn(batch_data)
                        loss = model(batch)
                        val_loss += loss.item()
                        val_step_count += 1
                    except Exception as e:
                        print(f"Validation error at step {val_step_count}: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_batches) if len(val_batches) > 0 else float('inf')
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {avg_val_loss}")
            wandb.log({'Val Loss': avg_val_loss})
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            
            print(f'Epoch {epoch} Val Loss {avg_val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')
            
            # 早停
            if epoch - best_epoch >= args.patience:
                print(f'Early stop at epoch {epoch}')
                break
    else:
        # 使用虚拟数据训练（备用方案）
        num_training_steps = args.num_epochs * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(args.num_epochs):
            # 训练
            model.train()
            epoch_loss = 0.0
            accum_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                try:
                    # 统一损失计算：loss = model(batch)
                    loss = model(batch)
                    loss.backward()
                    
                    clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    
                    if (step + 1) % args.grad_steps == 0:
                        adjust_learning_rate(optimizer.param_groups[0], args.lr, 
                                           step / len(train_loader) + epoch, epoch, args)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                    accum_loss += loss.item()
                    
                    if (step + 1) % args.grad_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        wandb.log({'Lr': lr})
                        wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                        accum_loss = 0.0
                    
                    progress_bar.update(1)
                    
                except Exception as e:
                    print(f"Training error at step {step}: {e}")
                    continue
            
            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss: {avg_train_loss}")
            wandb.log({'Train Loss (Epoch Mean)': avg_train_loss})
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    try:
                        loss = model(batch)
                        val_loss += loss.item()
                    except Exception as e:
                        print(f"Validation error at step {step}: {e}")
                        continue
            
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {avg_val_loss}")
            wandb.log({'Val Loss': avg_val_loss})
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            
            print(f'Epoch {epoch} Val Loss {avg_val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')
            
            # 早停
            if epoch - best_epoch >= args.patience:
                print(f'Early stop at epoch {epoch}')
                break
    
    # 清理内存
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    # 测试
    print("Starting evaluation...")
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    output_path = f'{args.output_dir}/{args.dataset}/unified_results_seed{args.seed}.json'
    
    model = _reload_best_model(model, args)
    model.eval()
    
    results = []
    with torch.no_grad():
        if kg_loader:
            # 使用真实数据测试
            for batch_data in test_batches:
                try:
                    batch = collate_fn(batch_data)
                    output = model.inference(batch)
                    for i, pred in enumerate(output['pred']):
                        result = {
                            'question': output['question'][i],
                            'prediction': pred,
                            'step': len(results)
                        }
                        results.append(result)
                except Exception as e:
                    print(f"Inference error: {e}")
                    continue
        else:
            # 使用虚拟数据测试
            for step, batch in enumerate(tqdm(test_loader)):
                try:
                    output = model.inference(batch)
                    for i, pred in enumerate(output['pred']):
                        result = {
                            'question': output['question'][i],
                            'prediction': pred,
                            'step': step * args.eval_batch_size + i
                        }
                        results.append(result)
                except Exception as e:
                    print(f"Inference error at step {step}: {e}")
                    continue
    
    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {output_path}')
    print(f'Total predictions: {len(results)}')
    
    wandb.log({'Test Samples': len(results)})


if __name__ == "__main__":
    args = parse_args()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()