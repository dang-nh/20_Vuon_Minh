export MASTER_PORT=29533
export WANDB_PROJECT=qwen3vl_tc
export NCCL_TIMEOUT=1800  # Keep timeout for safety
# --vit_lr 2e-6 \
# target q_proj k_proj v_proj o_proj out_proj gate_proj up_proj down_proj fc1 fc2
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
LOG_LEVEL=INFO \
MAX_PIXELS=802816 \
VIDEO_MAX_PIXELS=451584 \
swift sft \
    --tuner_backend peft \
    --target_modules all-linear \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --use_liger_kernel True \
    --dataset '/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_1.jsonl' \
    '/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_2.jsonl' \
    '/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_3.jsonl' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --packing False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --split_dataset_ratio 0 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --metric_for_best_model train_loss \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir /home/team_cv/tdkien/ms-swift/output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --dataset_num_proc 32 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --report_to wandb \
    --vit_lr 5e-7 \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --lora_bias "none" \
    --lr_scheduler_type "cosine" \
    --enable_dft_loss True \