torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir /home/hongchang/.llama/checkpoints/Llama-2-7b-chat/  \
    --tokenizer_path /home/hongchang/.llama/checkpoints/Llama-2-7b-chat/tokenizer.model \
    --max_seq_len 1024 --max_batch_size 8 \
    --dataset webqsp