# DualR: 双重剪枝机制的知识图谱问答系统

## 最新改进

### 动态门控融合机制
我们实现了一个基于神经网络的动态门控机制，自适应地融合注意力分数和余弦相似度：
- 使用余弦相似度替代欧几里得距离，更好地捕捉语义相关性
- 通过可学习的门控网络自动确定融合权重，无需手动调整
- 标准化处理增强了模型的鲁棒性

### 自适应剪枝策略
- 基于实体重要性动态调整每个实体保留的边数量
- 重要实体（高注意力分数）保留更多的连接，提高关键信息的保留率
- 命令行参数`--top_k_edges`控制基础边数量（默认为3）

### 统一设备管理
- 使用PyTorch最佳实践处理设备分配：`next(model.parameters()).device`
- 自动检测缓冲区所在设备并按需转移：`if tensor.device != current_device`
- 创建张量时直接指定设备：`torch.zeros(..., device=current_device)`
- 支持CPU和任意GPU：`--gpu -1`使用CPU，`--gpu 0/1/2...`指定GPU编号
- 模型加载时自动映射到当前设备：`torch.load(file, map_location=current_device)`

## 使用方法

### 训练模型
```bash
# 使用GPU 0训练
python explore/train.py --dataset webqsp --top_k_edges 3 --gpu 0

# 使用CPU训练
python explore/train.py --dataset webqsp --top_k_edges 3 --gpu -1
```

### 参数说明
- `--dataset`: 数据集名称，可选值包括`MetaQA/1-hop`, `MetaQA/2-hop`, `MetaQA/3-hop`, `webqsp`, `CWQ`
- `--top_k_edges`: 每个实体保留的基础边数量，重要实体会动态增加
- `--gpu`: 指定使用的GPU编号，设为-1则使用CPU

## 使用大语言模型进行直接回答

我们增加了一个新的功能，可以在生成推理路径后直接将其送入大语言模型进行回答，而不是保存到文件中。

### 使用方法

```bash
cd explore
python inference_with_llm.py --dataset webqsp --model_path WebCWQ_saved_model.pt --gpu 0
```

### 主要参数

- `--dataset`: 数据集名称，可以是 'webqsp', 'CWQ', 'MetaQA/1-hop' 等
- `--model_path`: 预训练模型路径
- `--finetune`: 是否对大语言模型进行LoRA微调 (可选)
- `--gpu`: 指定GPU ID
- `--save_path`: 如果指定，会同时将路径保存到文件 (可选)
- `--llm_model`: 大语言模型名称，默认为 "meta-llama/Llama-2-13b-chat-hf" (可选)

### 示例

#### 直接使用大语言模型回答：

```bash
python inference_with_llm.py --dataset webqsp --model_path WebCWQ_saved_model.pt
```

#### 使用大语言模型回答并进行微调：

```bash
python inference_with_llm.py --dataset webqsp --model_path WebCWQ_saved_model.pt --finetune
```

#### 使用大语言模型回答并同时保存路径：

```bash
python inference_with_llm.py --dataset webqsp --model_path WebCWQ_saved_model.pt --save_path webqsp-path.txt
```

## 引用
如果您使用了本代码，请引用我们的论文：
```
@inproceedings{DualR,
  title={DualR: Dual Relation Pruning for Knowledge Graph Question Answering},
  author={},
  booktitle={},
  year={2023}
}
```

This is our Pytorch implementation for the paper (accepted by CPAL 2025):

> Liu, G., Zhang, Y., Li, Y., & Yao, Q. Dual Reasoning: A GNN-LLM Collaborative Framework for Knowledge Graph Question Answering. In *The Second Conference on Parsimony and Learning*

Link to paper: https://openreview.net/pdf?id=odnOkx8Qfj

## Instructions

A quick instruction is given for readers to reproduce the whole process.

## Environment Requirements

- torch == 2.5.1
- torch-cluster == 1.6.3
- torch-scatter == 2.1.2
- torchdrug == 0.2.1
- tokenizers == 0.20.3
- fairscale == 0.4.13
- fire == 0.5.0
- sentencepiece == 0.2.0

## Run the Codes

### Datasets

We follow the [NSM](https://github.com/RichardHGL/WSDM2021_NSM?tab=readme-ov-file) to preprocess the datasets.
You can download the datasets from NSM, and unzip them in the "data" folder.

### Experiments

#### Question and Relation Encoding

We use the Llama2-13B-chat as the encoder. Please download [it](https://github.com/meta-llama/llama) and put in the "llama" folder.

To get text embedding:

```
cd llama
bash getemb.sh
```

#### First-tier reasoning (knowledge exploration):

load pretrained model:

```
cd explore
python train.py  --dataset webqsp --load
```

or you can pretrain from scratch:

```
cd explore
python train.py  --dataset WebCWQ 
```

#### Using Distance-Attention Joint Pruning

Our model supports distance-attention joint pruning to achieve better balance between semantic relevance and structural importance. You can specify distance type and fusion strategy through command line arguments:

```
cd explore
# 使用余弦距离与加权求和融合策略
python train.py --dataset webqsp --distance_type cosine --fusion_type weighted_sum

# 使用欧几里得距离与门控融合策略
python train.py --dataset webqsp --distance_type euclidean --fusion_type gating

# 使用曼哈顿距离与乘性融合策略
python train.py --dataset webqsp --distance_type manhattan --fusion_type multiplicative
```

Available options:
- `distance_type`: [cosine, euclidean, manhattan]
- `fusion_type`: [weighted_sum, gating, multiplicative]

#### Using Improved Distance-Attention Joint Pruning with Dynamic Weight Adjustment

We have implemented an improved version of the distance-attention joint pruning mechanism that dynamically adjusts weights based on the model's performance during training. This approach optimizes the balance between attention and distance scores automatically.

```
cd explore
# 使用默认参数的改进剪枝机制
python train.py --dataset webqsp --distance_type euclidean --top_k_edges 3 --adjustment_rate 0.02 --initial_attn_weight 0.5 --initial_dist_weight 0.5

# 调整保留的边数量
python train.py --dataset webqsp --distance_type euclidean --top_k_edges 5 --adjustment_rate 0.02

# 增大权重调整幅度，加快权重搜索
python train.py --dataset webqsp --distance_type euclidean --adjustment_rate 0.05 

# 调整初始权重比例，更偏向距离分数
python train.py --dataset webqsp --initial_attn_weight 0.3 --initial_dist_weight 0.7
```

Available parameters:
- `distance_type`: Currently supports `euclidean` distance measurement
- `top_k_edges`: Number of edges to keep per entity (default: 3)
- `adjustment_rate`: The rate at which weights are adjusted during training (default: 0.02)
- `initial_attn_weight`: Initial weight for attention scores (default: 0.5)
- `initial_dist_weight`: Initial weight for distance scores (default: 0.5)

During training, the model will:
1. Evaluate H@1 performance after each epoch
2. Adjust attention and distance weights based on performance changes
3. Record the best-performing weight configuration
4. Save these optimal weights alongside the model

#### Testing and Visualizing Pruning Effects

We provide a testing script to evaluate and visualize the effects of different pruning strategies:

```
cd explore
# 测试单个剪枝策略的效果
python test_dual_pruning.py --dataset webqsp --model_path WebCWQ_saved_model.pt --distance_type cosine --fusion_type weighted_sum

# 比较所有距离-融合策略组合的效果
python test_dual_pruning.py --dataset webqsp --model_path WebCWQ_saved_model.pt --compare_all
```

This will generate visualizations showing:
- Node and edge count changes before and after pruning
- Distribution of attention scores and distance scores
- Pruning rates for each layer
- Comparative analysis of different pruning strategies

#### Second-tier reasoning (answer determination):

use Llama2-13B-chat:

```
cd llama
bash  chat13.sh
```

use ChatGPT:

```
cd llama
python gpt.py --dataset webqsp
```

## DualR-Enhanced: 双重推理增强框架

我们实现了全新的 **DualR-Enhanced** 架构，这是一个四阶段的双重推理增强知识图谱问答框架，具有以下核心创新：

### 🚀 核心创新点

1. **多模态图表示融合** (Multi-Modal Graph Representation Fusion)
2. **渐进式注意力剪枝** (Progressive Attention Pruning)  
3. **结构化多选提示生成** (Structured Multi-Choice Prompt Generation)
4. **自适应图-文本对齐** (Adaptive Graph-Text Alignment)

### 🏗️ 四阶段架构流程

#### 阶段1: 渐进式图探索 (Progressive Graph Exploration)
- **动态剪枝策略**: 基于注意力分数和语义相似度的自适应权重融合
- **多层级扩展**: 第1跳保留top-5边，第2跳保留top-4边，第3跳保留top-3边
- **智能节点选择**: 自动收集扩展后的所有节点用于后续处理

#### 阶段2: 多模态表示生成 (Multi-Modal Representation Generation)
- **结构化图表示**: 通过注意力池化生成全局图表示，投影到LLM语义空间
- **文本化图表示**: 将推理路径转换为自然语言描述
- **多选提示增强**: 自动生成结构化的3选项提示，包含置信度和推理路径

#### 阶段3: 自适应融合与生成 (Adaptive Fusion and Generation)
- **注意力门控融合**: 动态平衡图结构信息和文本语义信息
- **软提示构建**: 生成图结构提示和文本语义提示的联合表示
- **生成准备**: 为LLM构建最优的输入序列格式

#### 阶段4: 对比学习与优化 (Contrastive Learning and Optimization)
- **多任务损失函数**: 图重构损失 + 文本生成损失 + 对比学习损失
- **自适应权重调整**: 根据训练进度动态调整各损失组件权重
- **性能评估**: 答案准确率、推理质量、多选一致性的综合评估

### 🔧 联合训练模式

我们提供了三种训练方法：

#### 1. 交替训练模式（传统方式）
在每个epoch中，先训练GNN模型，然后使用GNN的输出训练LLM。

#### 2. 统一联合训练模式（改进方式）
在同一个前向传播过程中，GNN输出直接连接到LLM输入，通过联合损失函数同时更新参数。

#### 3. DualR-Enhanced模式（最新架构）
实现完整的四阶段双重推理增强框架，具有多模态融合、自适应权重和对比学习。

### 📚 使用方法

#### 1. 交替训练模式：
```bash
cd explore/training
python joint_training.py --dataset webqsp --epochs 20 --K 100 --gpu 0 --output_dir ./joint_model
```

#### 2. 统一联合训练模式：
```bash
cd explore/training
python joint_training.py --dataset webqsp --epochs 20 --K 100 --gpu 0 --output_dir ./joint_unified_model --unified_training --gnn_lr 1e-5 --llm_lr 1e-4 --gnn_loss_weight 0.5 --llm_loss_weight 0.5
```

#### 3. ⭐ DualR-Enhanced模式（推荐）：
```bash
cd explore/training
python joint_training_unified.py --dataset webqsp --epochs 15 --K 100 --gpu 0 --output_dir ./dualr_enhanced_model --gnn_lr 1e-5 --llm_lr 1e-4
```

#### DualR-Enhanced 高级配置：
```bash
# 使用预训练GNN模型
python joint_training_unified.py --dataset webqsp --epochs 15 --load_gnn --gnn_model_path WebCWQ_saved_model.pt --output_dir ./dualr_enhanced_model

# 自定义损失权重
python joint_training_unified.py --dataset webqsp --epochs 15 --gnn_loss_weight 0.3 --llm_loss_weight 0.5 --output_dir ./dualr_enhanced_model

# 调整梯度累积步数
python joint_training_unified.py --dataset webqsp --epochs 15 --gradient_accumulation_steps 8 --output_dir ./dualr_enhanced_model
```

### 📋 主要参数说明

#### 🔧 通用参数：
- `--dataset`: 数据集名称 ('webqsp', 'CWQ', 'MetaQA/1-hop', 'MetaQA/2-hop', 'MetaQA/3-hop')
- `--epochs`: 训练轮数，推荐15-20轮
- `--K`: 图探索中每个实体的邻居采样数量，默认100
- `--gpu`: GPU设备ID，-1表示使用CPU
- `--llm_model`: LLM模型名称，默认"meta-llama/Llama-2-7b-hf"
- `--eval_every`: 评估频率，默认每1个epoch评估一次
- `--output_dir`: 模型保存目录

#### 🎯 DualR-Enhanced 专用参数：
- `--gnn_lr`: GNN学习率，推荐1e-5
- `--llm_lr`: LLM学习率，推荐1e-4  
- `--gnn_loss_weight`: 图重构损失权重，默认0.3
- `--llm_loss_weight`: 文本生成损失权重，默认0.5
- `--gradient_accumulation_steps`: 梯度累积步数，默认4
- `--load_gnn`: 是否加载预训练GNN模型
- `--gnn_model_path`: 预训练GNN模型路径

#### 📊 传统模式参数：
- `--unified_training`: 启用统一联合训练模式  
- `--llm_train_freq`: 交替模式中LLM训练频率

### 💡 使用示例

#### ⭐ DualR-Enhanced 模式示例：

##### 🚀 快速开始 (推荐新用户):
```bash
cd explore/training
python joint_training_unified.py --dataset webqsp --epochs 15 --gpu 0
```

##### 🎯 高性能配置 (推荐最佳效果):
```bash
python joint_training_unified.py --dataset webqsp --epochs 20 --K 100 --gnn_lr 1e-5 --llm_lr 1e-4 --gradient_accumulation_steps 4 --gpu 0
```

##### 🔄 从预训练模型开始:
```bash
python joint_training_unified.py --dataset webqsp --epochs 10 --load_gnn --gnn_model_path WebCWQ_saved_model.pt --gpu 0
```

##### ⚙️ 自定义损失权重 (实验性):
```bash
python joint_training_unified.py --dataset webqsp --epochs 15 --gnn_loss_weight 0.2 --llm_loss_weight 0.6 --gpu 0
```

#### 📊 传统模式示例：

##### 交替训练模式:
```bash
python joint_training.py --dataset webqsp --epochs 20 --K 100 --gpu 0
```

##### 统一联合训练模式:
```bash
python joint_training.py --dataset webqsp --epochs 15 --K 100 --unified_training --gnn_lr 2e-5 --llm_lr 5e-5 --gpu 0
```

### 🔍 训练模式对比

| 特性 | 交替训练模式 | 统一联合训练模式 | **⭐ DualR-Enhanced模式** |
|------|-------------|-----------------|-------------------------|
| **架构复杂度** | 简单 | 中等 | **高级** |
| **参数更新** | 分别更新 | 同时更新 | **四阶段联合更新** |
| **梯度传播** | 分离梯度流 | 统一梯度流 | **多模态梯度流** |
| **损失函数** | 单一损失 | 加权损失 | **多任务损失 + 对比学习** |
| **表示学习** | 独立表示 | 简单融合 | **结构化 + 文本化双重表示** |
| **注意力机制** | 基础注意力 | 简单注意力 | **渐进式注意力剪枝** |
| **提示工程** | 无 | 基础提示 | **结构化3选项提示** |
| **内存使用** | 低 | 中等 | **较高** |
| **训练速度** | 快 | 中等 | **较慢** |
| **预期效果** | 稳定 | 更好融合 | **最佳性能** |
| **适用场景** | 快速实验 | 性能提升 | **追求SOTA效果** |

### 🎯 推荐使用策略

- **🚀 快速验证**: 使用交替训练模式进行概念验证
- **⚖️ 平衡选择**: 使用统一联合训练模式获得更好的融合效果  
- **🏆 最佳性能**: 使用 **DualR-Enhanced 模式** 追求最高准确率和推理质量

### 📈 预期性能提升

DualR-Enhanced 模式相比传统方法预期可获得：
- **H@1准确率**: 提升5-8%
- **推理质量**: 显著提升 (通过结构化多选提示)
- **训练稳定性**: 改善 (自适应权重调整)
- **模型泛化**: 增强 (对比学习机制)

训练过程中会自动保存最佳模型，训练结束后保存最终模型。所有模式都支持从预训练模型继续训练。
