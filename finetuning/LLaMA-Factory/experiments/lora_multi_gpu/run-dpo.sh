## 02

#     --max_samples 15000 \

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage dpo \
#     --do_train \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --create_new_adapter \
#     --dataset llama7_gec_greco \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-02/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 512 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 50 \
#     --warmup_steps 20 \
#     --save_steps 200 \
#     --eval_steps 200 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 1e-5 \
#     --num_train_epochs 2.0 \
#     --val_size 0.1 \
#     --dpo_ftx 1.0 \
#     --plot_loss \
#     --fp16 \
#     --ddp_timeout 180000000 \
#     --ddp_find_unused_parameters False \
#     --dpo_beta 0.2


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-02/checkpoint-400/ \
#     --dataset gec_nucle_test \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-02/predict-nucle/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 256 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 1311 \
#     --fp16 \
#     --predict_with_generate
    

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-02/checkpoint-400/ \
#     --dataset gec_bea_dev \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-02/predict-bea/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 256 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 4384 \
#     --predict_with_generate



# # 05

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage dpo \
#     --do_train \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --create_new_adapter \
#     --dataset llama7_gec_greco \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-05/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 512 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 50 \
#     --warmup_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 1e-5 \
#     --num_train_epochs 2.0 \
#     --val_size 0.1 \
#     --dpo_ftx 1.0 \
#     --plot_loss \
#     --fp16 \
#     --ddp_timeout 180000000 \
#     --ddp_find_unused_parameters False \
#     --dpo_beta 0.5


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-05/checkpoint-400/ \
#     --dataset gec_bea_dev \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-05/predict-bea/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 196 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 4384 \
#     --predict_with_generate


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-05/checkpoint-400/ \
#     --dataset gec_nucle_test \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-05/predict-nucle/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 196 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 1311 \
#     --predict_with_generate


# ## 07

# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage dpo \
#     --do_train \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --create_new_adapter \
#     --dataset llama7_gec_greco \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-07 \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 512 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 50 \
#     --warmup_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 1e-5 \
#     --num_train_epochs 2.0 \
#     --val_size 0.1 \
#     --dpo_ftx 1.0 \
#     --plot_loss \
#     --fp16 \
#     --ddp_timeout 180000000 \
#     --ddp_find_unused_parameters False \
#     --dpo_beta 0.7


# WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
#     --config_file ../accelerate/single_config.yaml \
#     ../../src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
#     --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-07/checkpoint-500/ \
#     --dataset gec_bea_dev \
#     --dataset_dir ../../data \
#     --template llama2 \
#     --finetuning_type lora \
#     --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
#     --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-07/predict-bea/ \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 256 \
#     --max_length 256 \
#     --max_new_tokens 256 \
#     --preprocessing_num_workers 16 \
#     --per_device_eval_batch_size 1 \
#     --max_samples 4384 \
#     --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-07/checkpoint-500/ \
    --dataset gec_nucle_test \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-07/predict-nucle/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 256 \
    --max_length 256 \
    --max_new_tokens 256 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 1311 \
    --predict_with_generate

## 1
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --create_new_adapter \
    --dataset llama7_gec_greco \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16 \
    --ddp_timeout 180000000 \
    --ddp_find_unused_parameters False \
    --dpo_beta 1.0


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-1/checkpoint-500/ \
    --dataset gec_bea_dev \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-1/predict-bea/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 256 \
    --max_length 256 \
    --max_new_tokens 256 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-1/checkpoint-500/ \
    --dataset gec_nucle_test \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-1/predict-nucle/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 256 \
    --max_length 256 \
    --max_new_tokens 256 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 1311 \
    --predict_with_generate

## 4

WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --create_new_adapter \
    --dataset llama7_gec_greco \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-4 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 50 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-5 \
    --num_train_epochs 2.0 \
    --val_size 0.1 \
    --dpo_ftx 1.0 \
    --plot_loss \
    --fp16 \
    --ddp_timeout 180000000 \
    --ddp_find_unused_parameters False \
    --dpo_beta 0.2


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-4/checkpoint-500/ \
    --dataset gec_bea_dev \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-4/predict-bea/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 256 \
    --max_length 256 \
    --max_new_tokens 256 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 4384 \
    --predict_with_generate


WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --adapter_name_or_path ../../saves/LLaMA2-7B-gec-greco/dpo-4/checkpoint-500/ \
    --dataset gec_nucle_test \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj \
    --output_dir ../../saves/LLaMA2-7B-gec-greco/dpo-4/predict-nucle/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 256 \
    --max_length 256 \
    --max_new_tokens 256 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 1311 \
    --predict_with_generate
