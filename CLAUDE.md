# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DualR is a dual reasoning framework that combines Graph Neural Networks (GNN) and Large Language Models (LLM) for Knowledge Graph Question Answering (KGQA). The system implements a dual-tier approach: first-tier GNN reasoning for knowledge exploration and second-tier LLM reasoning for answer determination.

## Core Architecture

### Main Components

1. **GNN Model** (`explore/core/models.py`):
   - `Explore` class: Main GNN model with dynamic pruning mechanisms
   - `GNNLayer` class: Individual GNN layers with attention mechanisms
   - Supports dual pruning strategies (attention-based and distance-based)

2. **Base Model** (`explore/core/base_model.py`):
   - `BaseModel` class: Training and evaluation wrapper for the GNN model
   - Handles model training, validation, and path generation

3. **Data Loader** (`explore/data/load_data.py`):
   - `DataLoader` class: Handles data loading for MetaQA, WebQSP, and CWQ datasets
   - Manages knowledge graph construction and subgraph extraction

4. **LLM Integration** (`llama/` directory):
   - Llama2-13B integration for question and relation encoding
   - Text embedding generation and chat completion

## Common Commands

### Training Models

```bash
# Train GNN model on WebQSP dataset
cd explore
python train.py --dataset webqsp --K 100 --gpu 0

# Train with specific pruning parameters
python train.py --dataset webqsp --top_k_edges 3 --gpu 0

# Load pretrained model
python train.py --dataset webqsp --load --gpu 0
```

### Joint Training (GNN + LLM)

```bash
# Alternating training mode
cd explore
python joint_training.py --dataset webqsp --epochs 20 --K 100 --gpu 0

# Unified training mode (simultaneous parameter updates)
python joint_training.py --dataset webqsp --epochs 15 --K 100 --unified_training --gpu 0
```

### Generate Text Embeddings

```bash
cd llama
bash getemb.sh
```

### LLM Chat Completion

```bash
cd llama
bash chat13.sh
```

### Direct LLM Inference

```bash
cd explore
python inference_with_llm.py --dataset webqsp --model_path WebCWQ_saved_model.pt --gpu 0
```

## Dataset Support

- **MetaQA**: 1-hop, 2-hop, 3-hop versions
- **WebQSP**: Web Questions Semantic Parses
- **CWQ**: Complex Questions over Wikidata

## Key Features

### Dual Pruning Mechanisms

The system supports multiple pruning strategies:
- **Attention-based pruning**: Uses attention scores to select important edges
- **Distance-based pruning**: Uses cosine/euclidean distance for edge selection
- **Adaptive pruning**: Dynamically adjusts edge retention based on entity importance

### Device Management

- Supports both CPU and GPU training
- Use `--gpu -1` for CPU, `--gpu 0/1/2...` for specific GPU
- Automatic device detection and tensor migration

### Model Persistence

- GNN models saved as `.pt` files in `explore/models/`
- LLM models use HuggingFace format
- Results saved in `explore/results/` directory

## Development Notes

### Path Generation

The `visual_path` method in `Explore` class generates reasoning paths that can be:
- Saved to files (backward compatibility)
- Returned as structured data for LLM processing

### Joint Training Modes

1. **Alternating Training**: Train GNN and LLM separately in alternating epochs
2. **Unified Training**: Simultaneous parameter updates with joint loss function

### Configuration

Model hyperparameters are dataset-specific and defined in training scripts:
- Learning rates, decay rates, hidden dimensions
- Layer counts, dropout rates, batch sizes
- Pruning parameters (K, top_k_edges)

## Dependencies

Core requirements (from `llama/llama/requirements.txt`):
- torch == 2.5.1
- torch-cluster == 1.6.3
- torch-scatter == 2.1.2
- torchdrug == 0.2.1
- tokenizers == 0.20.3
- fairscale == 0.4.13
- fire == 0.5.0
- sentencepiece == 0.2.0

## File Structure

```
explore/
├── core/           # Core GNN models and base classes
├── data/           # Data loading and preprocessing
├── training/       # Training scripts and joint training
├── testing/        # Test scripts and evaluation
├── utils/          # Utility functions and helpers
├── models/         # Saved model files
└── results/        # Training results and performance logs

llama/
├── llama/          # Llama2 implementation
├── getemb.py       # Text embedding generation
├── gpt.py          # GPT integration
└── utils.py        # LLM utilities

data/
├── MetaQA/         # MetaQA dataset files
├── webqsp/         # WebQSP dataset files
└── CWQ/            # ComplexWebQuestions dataset files
```

## Testing

Run model evaluation:
```bash
cd explore
python test.py --dataset webqsp --model_path WebCWQ_saved_model.pt
```

Test dual pruning effects:
```bash
python test_dual_pruning.py --dataset webqsp --model_path WebCWQ_saved_model.pt --compare_all
```