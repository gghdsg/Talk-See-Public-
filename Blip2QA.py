from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import ToPILImage
import pandas as pd
import numpy as np
from tqdm import tqdm
from lavis.models import load_model
from lavis.models import load_model_and_preprocess

# 读取模型的方法
def load_and_prepare_model_QA(device):
    """加载并准备模型"""
    print(f"Loading Blip2 for QA Model...")
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    # prepare the image
    model.eval()
    model = model.to(device)
    return model, vis_processors  # 返回值是初始化（权重已经读取完毕的）Blip2


device = torch.device("cuda")
blip2_QAmodel, vis_preprocess = load_and_prepare_model_QA(device)


def blip2qa(image_path, question):
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_preprocess["eval"](raw_image).unsqueeze(0).to(device)
    answer = blip2_QAmodel.generate({"image": image, "prompt": "Question: "+ question})
    return answer[0]

def blip2qa_pil(pil_image, question):
    image = vis_preprocess["eval"](pil_image).unsqueeze(0).to(device)
    answer = blip2_QAmodel.generate({"image": image, "prompt": "Question: "+ question})
    return answer[0]

if __name__ == "__main__":
    raw_image = Image.open("2.png").convert("RGB")
#    image = vis_preprocess["eval"](raw_image).unsqueeze(0).to(device)
#    answer = blip2_QAmodel.generate({"image": image, "prompt": "Question: What's about this image ?"})
#    answer = blip2qa("2.png", "What's about this image ?")
    answer = blip2qa_pil(raw_image, "What's about this image ?")
    print(answer)



