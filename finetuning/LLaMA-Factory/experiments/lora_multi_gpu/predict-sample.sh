WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    ../../src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/oleksandr.korniienko/gec-llama2-7b-public/ \
    --dataset gec_bea_train \
    --dataset_dir ../../data \
    --template llama2 \
    --finetuning_type full \
    --output_dir ../../saves/LLaMA2-7B-chat-gec/full/1/ \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --max_length 512 \
    --max_new_tokens 512 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1 \
    --max_samples 34308 \
    --predict_with_generate \
    --do_sample True \
    --temperature 1.0 \
    --num_beams 5    
  