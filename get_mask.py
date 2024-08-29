from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from PIL import Image 
from gradio_demo.util_mask import get_mask_location
import torch 
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image 
import sys 
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

parsing_model = Parsing(0)
openpose_model = OpenPose(0)
openpose_model.preprocessor.body_estimation.model.to(device)

def get_mask(img_path):
    human_img = Image.open(img_path)
    human_img = human_img.resize((768,1024))
    keypoints = openpose_model(human_img.resize((384,512)))
    model_parse, _ = parsing_model(human_img.resize((384,512)))
    mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = mask.resize((768,1024))
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    return mask, mask_gray

if __name__ == '__main__':
    pic_path = sys.argv[1]
    mask_path = sys.argv[2]
    ano_path = sys.argv[3]
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    if not os.path.exists(ano_path):
        os.mkdir(ano_path)
    imgs = os.listdir(pic_path)
    for img in imgs:
        pic = os.path.join(pic_path, img)
        msk = os.path.join(mask_path, img)
        if os.path.exists(msk):
            continue
        ano = os.path.join(ano_path, img)
        if os.path.exists(ano):
            continue
        mask, mask_gray = get_mask(pic)
        mask.save(msk)
        mask_gray.save(ano)
