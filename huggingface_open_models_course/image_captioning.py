# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython"]
# ///

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForConditionalGeneration

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

from PIL import Image
import requests

raw_image =  Image.open(
    requests.get(img_url, stream=True).raw).convert('RGB')

raw_image.show()

text = "a photograph of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)

print(out)
print(processor.decode(out[0], skip_special_tokens=True))

def launch(pil_image):

    inputs = processor(pil_image,return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

import gradio as gr

iface = gr.Interface(launch, 
                     inputs=gr.Image(type='pil'), 
                     outputs=gr.Textbox(label='Caption'))

iface.launch(share=True, server_port=7831)
