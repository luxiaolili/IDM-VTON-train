CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision "fp16" test.py \
  --pretrained_model_name_or_path="idm" 