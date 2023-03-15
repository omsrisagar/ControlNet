import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
import os
from dataset import MyDataset
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", local_files_only=True)
# model = BlipForConditionalGeneration.from_pretrained("Salesfoce/blip-image-captioning-base", local_files_only=True).to("cuda")
processor = BlipProcessor.from_pretrained('/export/home/cuda00021/srikanth/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/snapshots/9f3084c0600e544506a2d414647492a036fb1aa0')
model = BlipForConditionalGeneration.from_pretrained('/export/home/cuda00021/srikanth/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base/snapshots/9f3084c0600e544506a2d414647492a036fb1aa0').to("cuda")

data_dir = '/export/home/cuda00022/srikanth/datasets/PITI_80k/train/train_img'
all_img_files = MyDataset._list_image_files_recursively(data_dir)
prompt_dict = {}
for img in tqdm(all_img_files):
    raw_image = Image.open(os.path.join(data_dir, img)).convert('RGB')

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    prompt_dict.update({img: caption})

save_fname = os.path.join(os.path.dirname(data_dir), 'prompt.json')
with open(save_fname, 'w') as f:
    json.dump(prompt_dict, f)

