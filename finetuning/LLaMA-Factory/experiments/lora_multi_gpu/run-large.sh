#!/bin/bash

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --dataset gec_dataset \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-13B-gec/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --warmup_steps 100 \
    --save_steps 200 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --max_steps 2000 \
    --val_size 0.001 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 \
    --quantization_bit 4

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path microsoft/phi-2 \
    --dataset gec_dataset \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/phi-2-gec/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --warmup_steps 100 \
    --save_steps 400 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --max_steps 2000 \
    --val_size 0.001 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16 \
    --quantization_bit 4


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
#     --dataset gec_dataset \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/Mixtral-8x7B-Instruct-gec/lora/sft \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 100 \
#     --warmup_steps 120 \
#     --save_steps 400 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --max_samples 6000 \
#     --val_size 0.1 \
#     --ddp_timeout 180000000 \
#     --plot_loss \
#     --fp16


