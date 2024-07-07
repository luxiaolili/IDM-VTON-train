import json 
import os
import pickle
from pathlib import Path
from typing import Tuple, Literal

import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from transformers import CLIPImageProcessor, AutoTokenizer
import torch.nn.functional as F

class VitonHDDataset(data.Dataset):
    def __init__(self,
        dataroot_path,
        phase,
        caption_filename,
        size,
        tokenizer_1,
        tokenizer_2,
        ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.caption_filename = caption_filename
        self.category = ('upper_body')
        self.height = size[0]
        self.width = size[1]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )   
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.toTensor = transforms.ToTensor()
        self.clip_processor = CLIPImageProcessor()
        self.outputs = []
        self.cloth_caption = self.get_label(self.caption_filename, self.phase)
        self.data_pair = self.get_pair(self.phase + '_pairs.txt', self.phase)

    def get_pair(self, data_file, phase):
        person_cloth = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            pic_name = line.split(' ')[0]
            cloth_name = line.split(' ')[1]
            if phase == 'train':
                cloth_name = pic_name
            person_cloth[pic_name] = cloth_name
        return person_cloth

     
    def __len__(self):
        return len(self.cloth_caption)
    
    def __getitem__(self, index):
        phase=self.phase
        base_path = os.path.join(self.dataroot, phase)
        cloth_path = os.path.join(base_path, 'cloth')
        img_path = os.path.join(base_path, 'image')
        agno_path = os.path.join(base_path, 'agnostic-v3.3')
        img_mask_path = os.path.join(base_path, 'image_mask')
        densepose_path = os.path.join(base_path, 'image-densepose')
        img_label = self.cloth_caption[index]
        key = img_label['image_file']
        pic_name = key + "_00.jpg"
        text = random.choice(img_label['text'])
        if phase == 'train':
            cloth_name = pic_name
        else:
            cloth_name = self.data_pair[pic_name]
        cloth_img_path = os.path.join(cloth_path, cloth_name)
        person_img_path = os.path.join(img_path, pic_name)
        agno_img_path = os.path.join(agno_path, pic_name)
        densepose_img_path = os.path.join(densepose_path, pic_name)
        cloth_mask_path = os.path.join(img_mask_path, pic_name)
        
        sample = {}
        sample['im_name'] = pic_name
        sample["cloth_pure"] = self.transform(Image.open(cloth_img_path).resize((self.width, self.height)))
        sample['cloth'] = self.clip_processor(images=Image.open(cloth_img_path), return_tensors="pt").pixel_values
        cloth_caption= 'A photo of ' + text
        sample['cloth_caption'] = cloth_caption
        sample['caption_cloth_input_ids'] = self.tokenizer_1(cloth_caption,max_length=self.tokenizer_1.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids
        sample['caption_cloth_input_ids_2'] = self.tokenizer_2(cloth_caption,max_length=self.tokenizer_2.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids
        sample["image"] = self.transform(Image.open(person_img_path).resize((self.width, self.height)))
        caption = 'Model is wearing ' + text
        sample['caption'] = caption
        sample['caption_input_ids'] = self.tokenizer_1(caption,max_length=self.tokenizer_1.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids
        sample['caption_input_ids_2'] = self.tokenizer_2(caption,max_length=self.tokenizer_2.model_max_length,padding="max_length",truncation=True,return_tensors="pt").input_ids
        sample['denspose'] = self.transform(Image.open(densepose_img_path).resize((self.width, self.height)))
        mask = Image.open(cloth_mask_path).convert('L')
        mask = self.toTensor(mask)
        mask = mask.detach().numpy()
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)
        sample['agno'] = sample['image'] * (mask < 0.5)
        sample['mask'] = mask
        sample['original_size'] = torch.tensor([self.height, self.width])
        sample['crop_coords_top_left'] = torch.tensor([0, 0])
        sample['target_size'] = torch.tensor([self.height, self.width])
        return sample
        

    def get_label(self, caption_filename, phase):
        data_list = []
        with open(caption_filename, 'r') as f:
            vot_dict = json.load(f)
        base_path = os.path.join(self.dataroot, phase)
        img_path = os.path.join(base_path, 'image')
        for key in vot_dict.keys():
            img_label = {}
            pic_name = key + "_00.jpg"
            im_path = os.path.join(img_path, pic_name)
            if os.path.exists(im_path):
                img_label['image_file'] = key
                img_label['text'] = []
                for i in range (len(vot_dict[key])):
                    img_label['text'].append(vot_dict[key][i])
                data_list.append(img_label)
        return data_list

if __name__ == '__main__':
    tokenizer_one = AutoTokenizer.from_pretrained(
        "idm",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        "idm",
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    dataroot_path = '../zalando-hd-resized'
    phase='test'
    caption_filename = 'vitonhd.json'
    size = (1024, 768)
    dataset = VitonHDDataset(dataroot_path, phase, caption_filename,size,tokenizer_one, tokenizer_two, )
    train_dataloader = torch.utils.data.DataLoader(dataset,shuffle=True,batch_size=1,num_workers=4)
    for step, sample in enumerate(train_dataloader):
        print(sample['agno'].shape, sample['mask'].shape, sample['image'].shape)
        break