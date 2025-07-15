# DualR: Dual Reasoning Framework for Knowledge Graph Question Answering

DualR is a dual reasoning framework that combines Graph Neural Networks (GNN) and Large Language Models (LLM) for Knowledge Graph Question Answering (KGQA). The system implements a dual-tier approach: first-tier GNN reasoning for knowledge exploration and second-tier LLM reasoning for answer determination.

## 📋 Overview

The framework consists of:
- **GNN Model**: Dynamic pruning mechanisms with dual subgraphs (Top-5 and Top-3)
- **LLM Integration**: Llama2-13B with 8-bit quantization for efficiency
- **Unified Training**: End-to-end training with single loss function
- **Multi-Choice Reasoning**: Candidate generation with reasoning chains

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch==2.5.1 torch-cluster==1.6.3 torch-scatter==2.1.2
pip install transformers==4.35.0 bitsandbytes
pip install wandb tqdm pandas numpy
pip install peft fairscale fire sentencepiece
```

### Data Preparation

1. Download datasets to `data/` directory:
   - **MetaQA**: `data/MetaQA/1-hop/`, `data/MetaQA/2-hop/`, `data/MetaQA/3-hop/`
   - **WebQSP**: `data/webqsp/`
   - **CWQ**: `data/CWQ/`

2. Download pre-computed embeddings to `embedding/` directory:
   - Question embeddings: `Meta-*-train/valid/test.npy`, `webqsp-*.npy`, `CWQ-*.npy`
   - Relation embeddings: `Meta-rel.npy`, `webqsp-rel.npy`, `CWQ-rel.npy`

## 🔧 Training

### Option 1: Traditional GNN Training

```bash
cd explore

# Train GNN model on WebQSP dataset
python train.py --dataset webqsp --K 100 --gpu 0

# Train with specific pruning parameters
python train.py --dataset webqsp --top_k_edges 3 --gpu 0

# Load pretrained model
python train.py --dataset webqsp --load --gpu 0
```

### Option 2: Joint Training (GNN + LLM)

```bash
cd explore

# Alternating training mode
python joint_training.py --dataset webqsp --epochs 20 --K 100 --gpu 0

# Unified training mode (recommended)
python joint_training.py --dataset webqsp --epochs 15 --K 100 --unified_training --gpu 0
```

### Option 3: Unified Training Architecture (Latest)

```bash
cd explore

# Unified end-to-end training with single loss function
python train_unified.py --dataset webqsp \
                       --llm_model_path meta-llama/Llama-2-7b-chat-hf \
                       --llm_frozen True \
                       --batch_size 2 \
                       --num_epochs 10 \
                       --lr 5e-5 \
                       --gpu 0

# Training with LoRA fine-tuning
python train_unified.py --dataset webqsp \
                       --llm_frozen False \
                       --batch_size 1 \
                       --num_epochs 15 \
                       --lr 1e-4 \
                       --gpu 0
```

## 📊 Training Parameters

### GNN Parameters
- `--hidden_dim`: Hidden dimension (default: 200)
- `--attn_dim`: Attention dimension (default: 200)
- `--n_layer`: Number of GNN layers (default: 3)
- `--K`: Subgraph size (default: 100)
- `--sample`: Sampling flag (default: 1)
- `--dropout`: Dropout rate (default: 0.1)

### LLM Parameters
- `--llm_model_path`: LLM model path (default: meta-llama/Llama-2-7b-chat-hf)
- `--llm_frozen`: Whether to freeze LLM (default: True)
- `--max_txt_len`: Maximum text length (default: 512)
- `--max_new_tokens`: Maximum generation tokens (default: 64)

### Training Parameters
- `--batch_size`: Training batch size (default: 2)
- `--eval_batch_size`: Evaluation batch size (default: 4)
- `--num_epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 5e-5)
- `--patience`: Early stopping patience (default: 3)

## 🎯 Key Features

### Dual Pruning Mechanisms
- **Attention-based pruning**: Uses attention scores to select important edges
- **Distance-based pruning**: Uses cosine/euclidean distance for edge selection
- **Adaptive pruning**: Dynamically adjusts edge retention based on entity importance

### Unified Training Architecture
- **Single Loss Function**: `loss = model(batch)` - no multi-loss combination
- **End-to-End Training**: Gradients flow through both GNN and LLM components
- **Soft Prompt Mechanism**: `soft_prompt = [h_g ; MEAN(h_t)]`
- **8-bit Quantization**: Reduces LLM memory usage by ~50%

### CSV-based Graph Textualization
```
node_name
Barack Obama
Hawaii
USA

src,edge_attr,dst
Barack Obama,born_in,Hawaii
Hawaii,located_in,USA
```

## 💾 Model Outputs

### Training Results
- Checkpoints saved in `explore/results/{dataset}/`
- Best model: `best_model.pt`
- Training logs: WandB integration

### Inference Results
- JSON format with predictions
- Reasoning chains for multi-choice answers
- Performance metrics

## 🔬 Inference

### Direct LLM Inference
```bash
cd explore
python inference_with_llm.py --dataset webqsp \
                             --model_path WebCWQ_saved_model.pt \
                             --gpu 0
```

### Model Evaluation
```bash
cd explore
python test.py --dataset webqsp --model_path WebCWQ_saved_model.pt
```

## 📈 Performance

The framework supports evaluation on:
- **MetaQA**: 1-hop, 2-hop, 3-hop versions
- **WebQSP**: Web Questions Semantic Parses  
- **CWQ**: Complex Questions over Wikidata

## 🔧 Troubleshooting

### Memory Issues
- Reduce `--batch_size` to 1
- Use `--llm_frozen True` to freeze LLM parameters
- Enable 8-bit quantization (already configured)

### Data Loading Errors
- Check data paths in `data/` directory
- Verify embedding files in `embedding/` directory
- Ensure correct dataset format

### GPU Issues
- Use `--gpu -1` for CPU training
- Check CUDA compatibility
- Reduce model size with quantization

## 📝 Model Architecture

```
Input Question + Subgraph
         ↓
    GNN Encoding (3 layers)
    ├── Top-5 Subgraph A → Textualization → h_t
    └── Top-3 Subgraph B → Structure Vector → h_g
         ↓
    Soft Prompt = [h_g ; MEAN(h_t)]
         ↓
    LLM Generation (Llama2-7B + 8bit)
         ↓
    Answer Generation
```

## 🤝 Contributing

1. Follow the existing code style
2. Test on all three datasets
3. Update documentation
4. Submit pull requests

## 📄 License

This project is licensed under the MIT License.