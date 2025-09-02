"""
GNN-LLM训练配置文件
包含不同数据集的推荐参数设置
"""

# 不同数据集的推荐配置
DATASET_CONFIGS = {
    "webqsp": {
        # GNN参数
        "hidden_dim": 256,
        "attn_dim": 5,
        "n_layer": 3,
        "K": 150,
        "dropout": 0.1,
        "act": "relu",
        
        # 训练参数
        "batch_size": 2,
        "eval_batch_size": 4,
        "num_epochs": 15,
        "lr": 1e-4,
        "wd": 0.01,
        "patience": 5,
        "warmup_epochs": 1,
        
        # LLM参数
        "max_txt_len": 512,
        "max_new_tokens": 256,
        "llm_frozen": "False",
    },
    
    "CWQ": {
        # GNN参数
        "hidden_dim": 256,
        "attn_dim": 5,
        "n_layer": 3,
        "K": 100,
        "dropout": 0.1,
        "act": "relu",
        
        # 训练参数
        "batch_size": 1,  # CWQ问题较复杂，使用较小batch size
        "eval_batch_size": 2,
        "num_epochs": 20,
        "lr": 5e-5,  # 较小的学习率
        "wd": 0.01,
        "patience": 8,
        "warmup_epochs": 2,
        
        # LLM参数
        "max_txt_len": 768,  # CWQ需要更长的文本
        "max_new_tokens": 256,
        "llm_frozen": "False",
    },
    
    "MetaQA/1-hop": {
        # GNN参数
        "hidden_dim": 256,
        "attn_dim": 5,
        "n_layer": 1,  # 1-hop任务较简单
        "K": 40,
        "dropout": 0.1,
        "act": "idd",
        
        # 训练参数
        "batch_size": 4,
        "eval_batch_size": 8,
        "num_epochs": 10,
        "lr": 2e-4,
        "wd": 0.01,
        "patience": 3,
        "warmup_epochs": 1,
        
        # LLM参数
        "max_txt_len": 256,  # MetaQA问题较短
        "max_new_tokens": 128,
        "llm_frozen": "False",
    },
    
    "MetaQA/2-hop": {
        # GNN参数
        "hidden_dim": 256,
        "attn_dim": 5,
        "n_layer": 2,
        "K": 50,
        "dropout": 0.1,
        "act": "idd",
        
        # 训练参数
        "batch_size": 3,
        "eval_batch_size": 6,
        "num_epochs": 12,
        "lr": 1.5e-4,
        "wd": 0.01,
        "patience": 4,
        "warmup_epochs": 1,
        
        # LLM参数
        "max_txt_len": 384,
        "max_new_tokens": 128,
        "llm_frozen": "False",
    },
    
    "MetaQA/3-hop": {
        # GNN参数
        "hidden_dim": 256,
        "attn_dim": 5,
        "n_layer": 3,
        "K": 60,
        "dropout": 0.1,
        "act": "idd",
        
        # 训练参数
        "batch_size": 2,
        "eval_batch_size": 4,
        "num_epochs": 15,
        "lr": 1e-4,
        "wd": 0.01,
        "patience": 5,
        "warmup_epochs": 1,
        
        # LLM参数
        "max_txt_len": 512,
        "max_new_tokens": 128,
        "llm_frozen": "False",
    }
}

# LLM模型路径配置
LLM_MODEL_PATHS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "vicuna-13b": "lmsys/vicuna-13b-v1.5",
}

# LoRA配置
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

def get_config(dataset_name):
    """获取指定数据集的配置"""
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]
    else:
        print(f"Warning: No specific config for {dataset_name}, using webqsp config")
        return DATASET_CONFIGS["webqsp"]

def print_config(dataset_name):
    """打印指定数据集的配置"""
    config = get_config(dataset_name)
    print(f"Configuration for {dataset_name}:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)

if __name__ == "__main__":
    # 示例：打印所有数据集的配置
    for dataset in DATASET_CONFIGS.keys():
        print_config(dataset)
        print()