#!/bin/bash



# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-chat-gec/lora-all/sft/checkpoint-1200/ \
#     --dataset gec_dataset_bea_test \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/LLaMA2-7B-chat-gec/lora-all-bea/predict \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 4 \
#     --max_samples 4384 \
#     --predict_with_generate



WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B-chat-gec/lora-all-6k/sft/checkpoint-2800/ \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-chat-gec/lora-all-6k/predict-2800/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 4 \
    --max_samples 4384 \
    --predict_with_generate

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path google/gemma-7b-it \
#     --adapter_name_or_path ../../saves/Gemma-7B-chat-gec/lora/sft \
#     --dataset gec_dataset_test \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/Gemma-7B-chat-gec/lora/predict \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 1311 \
#     --predict_with_generate

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --adapter_name_or_path ../../saves/LLaMA2-7B-chat-gec/lora/sft/checkpoint-1200/ \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/LLaMA2-7B-chat-gec/lora/stft/checkpoint-1200/predict/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 4 \
    --max_samples 4384 \
    --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path google/gemma-7b-it \
    --adapter_name_or_path ../../saves/Gemma-7B-chat-gec/lora/sft/checkpoint-1200 \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/Gemma-7B-chat-gec/lora/predict-bea \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path google/gemma-7b-it \
    --adapter_name_or_path ../../saves/Gemma-7B-chat-gec-3/lora-all/sft/checkpoint-2000 \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/Gemma-7B-chat-gec-3/lora-all/sft/checkpoint-2000/predict-bea \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
#     --adapter_name_or_path ../../saves/Mistral-7B-Instruct-gec/lora/sft \
#     --dataset gec_dataset_test \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/Mistral-7B-Instruct-gec/lora/predict \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 1311 \
#     --predict_with_generate


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path microsoft/phi-2 \
#     --adapter_name_or_path ../../saves/phi-2-gec/lora/sft \
#     --dataset gec_dataset_test \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/phi-2-gec/lora/predict \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 1311 \
#     --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path google/gemma-7b-it \
    --adapter_name_or_path ../../saves/Gemma-7B-chat-gec-3/lora-all/sft/checkpoint-2000 \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,o_proj,k_proj,v_proj,gate_proj,up_proj,down_proj \
    --output_dir ../../saves/Gemma-7B-chat-gec-3/lora-all/sft/checkpoint-2000/predict-bea \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate



WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
    --adapter_name_or_path ../../saves/Mistral-7B-Instruct-gec-2/lora-all/sft/checkpoint-2000 \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ../../saves/Mistral-7B-Instruct-gec-2/lora-all/sft/checkpoint-2000/predict-bea \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 python ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path google/gemma-7b-it \
    --adapter_name_or_path ../../saves/Mistral-7B-Instruct-gec-2/lora-all/sft/checkpoint-2000 \
    --dataset gec_dataset_bea_test \
    --dataset_dir ../../data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/Mistral-7B-Instruct-gec-2/lora-all/sft/checkpoint-2000/predict-bea \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate