OMP_NUM_THREADS=14 accelerate launch --config_file "configs/fsdp_config_allparams.yaml" train.py \
--seed 100 \
--model_name_or_path "/local_disk0/llama-3.1-8b-instruct" \
--dataset_path "/dbfs/ml/augmented_ITAR_SFT" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 2048 \
--num_train_epochs 13 \
--logging_steps 10 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "steps" \
--save_steps 1 \
--bf16 True \
--packing False \
--learning_rate 5e-5 \
--lr_scheduler_type "linear" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/dbfs/ml/ITAR_qa_full_llama3.1_8b" \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_checkpointing True \
--dataset_text_field "content" \
--use_flash_attn False \
--use_peft_lora False \
--report_to "none"
