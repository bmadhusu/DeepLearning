# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython"]
# ///

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import BlipForImageTextRetrieval
model = BlipForImageTextRetrieval.from_pretrained(
    "Salesforce/blip-itm-base-coco")


from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-itm-base-coco")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

from PIL import Image
import requests

raw_image =  Image.open(
    requests.get(img_url, stream=True).raw).convert('RGB')

raw_image.show()

text = "an image of a woman and a dog on the beach"

inputs = processor(images=raw_image,
                   text=text,
                   return_tensors="pt")

itm_scores = model(**inputs)[0]

print(itm_scores)

import torch

itm_score = torch.nn.functional.softmax(
    itm_scores,dim=1)

print(f"""\
The image and text are matched \
with a probability of {itm_score[0][1]:.4f}""")

