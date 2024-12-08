export MODEL_NAME="/openbayes/home/checkpoints/stable-diffusion-v1-4"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --validation_prompts "yoda" "sasuke" "doraemon" "corgi" \
  --validation_epochs=10 \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --checkpointing_steps=5000 \
  --checkpoints_total_limit 1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model" \
  --report_to "wandb" \
  --resume_from_checkpoint latest