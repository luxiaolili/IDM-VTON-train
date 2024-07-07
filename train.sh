CUDA_VISIBLE_DEVICES=1  accelerate launch --num_processes 1 --mixed_precision "fp16" train.py \
  --pretrained_model_name_or_path="idm" \
  --inpainting_model_path="stable-diffusion-xl-1.0-inpainting-0.1" \
  --garmnet_model_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --width=768 \
  --height=1024 \
  --data_json_file="vitonhd.json" \
  --data_root_path="../zalando-hd-resized" \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --dataloader_num_workers=6 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --resume_from_checkpoint="checkpoint-350000" \
  --output_dir="idm_plus_output_up"\
  --save_steps=50000