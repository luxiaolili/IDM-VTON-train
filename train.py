# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
import math
import safetensors
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available

#from diffusers import UNet2DConditionModel
from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

from vitonhd_dataset import VitonHDDataset
import time
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


logger = get_logger(__name__, log_level="INFO")



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "idm",required=False,)
    parser.add_argument("--inpainting_model_path",type=str,default= "idm",required=False,)
    parser.add_argument("--garmnet_model_path",type=str,default= "idm",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="result",)
    parser.add_argument("--logging_dir",type=str,default="logs",)
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="Learning rate to use.",)
    parser.add_argument("--scale_lr",action="store_true",default=False,help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--data_dir",type=str,default="../zalando-hd-resized")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--report_to",type=str,default="tensorboard",help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"`'' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
    parser.add_argument("--data_root_path",type=str,default="",required=True,help="Training data root path",)
    parser.add_argument("--data_json_file",type=str,default=None,required=True,help="data label",)
    parser.add_argument("--train_batch_size", type=int, default=1,)
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument("--dataloader_num_workers",type=int,default=0,)
    parser.add_argument("--save_steps",type=int,default=2000,help=("Save a checkpoint of the training state every X updates"),)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--lr_scheduler",type=str,default="constant", help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--use_8bit_adam", action="store_true",default=False, help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--snr_gamma",type=float,default=None,help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. ""More details here: https://arxiv.org/abs/2303.09556.",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--checkpointing_steps",type=int,default=500,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"" training using `--resume_from_checkpoint`."),)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),)
    parser.add_argument("--validation_epochs",type=int,default=5,help="Run validation every X epochs.",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()

    return args


def compute_vae_encodings(pixel_values, vae):
    
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return model_input

def load_model_with_zeroed_mismatched_keys(unet, pretrained_weights_path):
    # Load the pretrained weights
    # new weight as an initialized state
    # Determine the file type and load the pretrained weights
    if pretrained_weights_path.endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(pretrained_weights_path)
    elif pretrained_weights_path.endswith('.bin'):
        state_dict = torch.load(pretrained_weights_path)
    else:
        raise ValueError("Unsupported file type. Only .safetensors and .bin are supported.")
    
    # Initialize a new state dict for the model
    # The new unet data structure
    new_state_dict = unet.state_dict()
    
    # Iterate through the pretrained weights
    for key, value in state_dict.items():
        if key in new_state_dict and new_state_dict[key].shape == value.shape:
            new_state_dict[key] = value
        else:
            if key in unet.attn_processors.keys():
                # Initialize the ip adaptor for the model check https://github.com/tencent-ailab/IP-Adapter/blob/cfcdf8ce36f31e3d358b3c4c4b1bb78eab2854bd/tutorial_train_plus.py#L335
                layer_name = key.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet[layer_name + ".to_v.weight"],
                }
                new_state_dict[key] = weights
            else:
                print(f"Key {key} mismatched or not found in model. Initializing with zeros.")
                new_state_dict[key] = torch.zeros_like(new_state_dict[key])
    
    # Load the new state dict into the model
    unet.load_state_dict(new_state_dict)

def tokenize_caption(text, tokenizer_1, tokenizer_2):
    text_inputs = tokenizer_1(
            text, max_length=tokenizer_1.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
    text_inputs_2 = tokenizer_2(
            text, max_length=tokenizer_2.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
    return text_inputs.input_ids, text_inputs_2.input_ids


def collate_fn(data):
    cloth_pure = torch.stack([example["cloth_pure"] for example in data])
    cloth = torch.cat([example["cloth"] for example in data], dim=0)
    caption_cloth_input_ids = torch.cat([example["caption_cloth_input_ids"] for example in data], dim=0)
    caption_cloth_input_ids_2 = torch.cat([example["caption_cloth_input_ids_2"] for example in data], dim=0)
    image = torch.stack([example["image"] for example in data])
    caption_input_ids = torch.cat([example["caption_input_ids"] for example in data], dim=0)
    caption_input_ids_2 = torch.cat([example["caption_input_ids_2"] for example in data], dim=0)
    agno = torch.stack([example["agno"] for example in data])
    denspose = torch.stack([example["denspose"] for example in data])
    mask =  torch.stack([example["mask"] for example in data])
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    return {
        "cloth_pure": cloth_pure,
        "cloth": cloth,
        "caption_cloth_input_ids": caption_cloth_input_ids,
        "caption_cloth_input_ids_2": caption_cloth_input_ids_2,
        "image": image,
        "caption_input_ids": caption_input_ids,
        "caption_input_ids_2": caption_input_ids_2,
        "agno": agno,
        "denspose": denspose,
        "mask": mask,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
    }

def main():
    args = parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        'madebyollin',
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.inpainting_model_path,
        subfolder="unet",
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True
    )
    pretrain = os.path.join(args.inpainting_model_path, 'unet')
    load_model_with_zeroed_mismatched_keys(unet, os.path.join(pretrain, 'diffusion_pytorch_model.safetensors'))
    if unet.conv_in.in_channels == 9:
        logger.info("Initializing the Inpainting UNet from the pretrained 9 channel UNet .")
        in_channels = 13
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :9, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
    )

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        args.garmnet_model_path,
        subfolder="unet",
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
 

    UNet_Encoder.to(accelerator.device, weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, weight_dtype)
    text_encoder_two.to(accelerator.device, weight_dtype)
    image_encoder.to(accelerator.device, weight_dtype)
    unet.train()
    UNet_Encoder.eval()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    train_dataset = VitonHDDataset(
        dataroot_path=args.data_root_path,
        phase="train",
        caption_filename=args.data_json_file,
        tokenizer_1=tokenizer_one,
        tokenizer_2=tokenizer_two,
        size=(args.height, args.width),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=6,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, args.lr_scheduler
    )

    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    mask_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        begin = time.perf_counter()
        for step, sample in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    init_image = (sample['image'] + 1.0) /2.0
                    init_image = image_processor.preprocess(init_image, height=args.height, width=args.width, crops_coords=None, resize_mode="default"
                )
                    latents = compute_vae_encodings(init_image, vae)
                    cloth_latents = compute_vae_encodings(sample['cloth_pure'], vae)
                    pose_latents = compute_vae_encodings(sample['denspose'], vae)
                    mask_image_latents = compute_vae_encodings(sample['agno'],vae)
                    
                    mask = F.interpolate(sample['mask'].to(torch.float32), size=(int(args.height/8), int(args.width/8)))
                    mask = mask.to(accelerator.device, dtype=torch.float16)
                   
                    
                    latents = latents.to(accelerator.device, dtype=weight_dtype)
                    cloth_latents = cloth_latents.to(accelerator.device, dtype=weight_dtype)
                    pose_latents = pose_latents.to(accelerator.device, dtype=weight_dtype)
                    mask_image_latents = mask_image_latents.to(accelerator.device, dtype=weight_dtype)

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                bsz = latents.shape[0]
                
                input_latents = torch.cat([noisy_latents, mask, mask_image_latents, pose_latents], dim=1)
                input_latents = input_latents.to(accelerator.device, dtype=weight_dtype)
                

                # cloth_inputs_ids, cloth_inputs_2_ids = tokenize_caption(sample['caption_cloth'], tokenizer_one, tokenizer_two)
                # text_inputs_ids, text_inputs_2_ids = tokenize_caption(sample['caption'], tokenizer_one, tokenizer_two)

                

                with torch.no_grad():
                    cloth_encoder_output = text_encoder_one(sample['caption_cloth_input_ids'].to(accelerator.device), output_hidden_states=True)
                    cloth_text_embeds = cloth_encoder_output.hidden_states[-2]
                    cloth_encoder_output_2 = text_encoder_two(sample['caption_cloth_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    cloth_pooled_text_embeds = cloth_encoder_output_2[0]
                    cloth_text_embeds_2 = cloth_encoder_output_2.hidden_states[-2]
                    cloth_text_embeds = torch.concat([cloth_text_embeds, cloth_text_embeds_2], dim=-1) # concat

                    encoder_output = text_encoder_one(sample['caption_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_two(sample['caption_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat

                

                # add cond
                add_time_ids = [
                    sample["original_size"].to(accelerator.device),
                    sample["crop_coords_top_left"].to(accelerator.device),
                    sample["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                added_cond_kwargs = {"text_embeds": cloth_pooled_text_embeds, "time_ids": add_time_ids}
                
                with torch.no_grad():
                    #cloth_image_embeds = image_encoder(sample["cloth"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    cloth_image_embeds = image_encoder(sample["cloth"].to(accelerator.device, dtype=weight_dtype),output_hidden_states=True).hidden_states[-2]
                    cloth_image_embeds = cloth_image_embeds.repeat_interleave(1, dim=0)
                    
                cloth_image_embeds = unet.encoder_hid_proj(cloth_image_embeds.to(accelerator.device, dtype=torch.float32))
                #cloth_image_embeds = unet.module.encoder_hid_proj(cloth_image_embeds.to(accelerator.device, dtype=torch.float32))
                cloth_image_embeds = cloth_image_embeds.to(accelerator.device, dtype=weight_dtype)
                down,reference_features = UNet_Encoder(cloth_latents, timesteps, cloth_text_embeds,return_dict=False)
                reference_features = list(reference_features)
                
                added_cond_kwargs["image_embeds"] = cloth_image_embeds
                cross_attention_kwargs = None
                noise_pred = unet(input_latents, timesteps, encoder_hidden_states=text_embeds,added_cond_kwargs=added_cond_kwargs, return_dict=False, garment_features=reference_features)[0]
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, lr: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, args.learning_rate))
                
            global_step += 1 
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)


if __name__ == "__main__":
    main()
