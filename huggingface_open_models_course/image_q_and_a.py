# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython"]
# ///

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

from transformers import BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-vqa-base")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

from PIL import Image
import requests

raw_image =  Image.open(
    requests.get(img_url, stream=True).raw).convert('RGB')

raw_image.show()

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt")
out = model.generate(**inputs)

print(out)
print(processor.decode(out[0], skip_special_tokens=True))

def launch(pil_image, question):

    inputs = processor(pil_image, question, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

import gradio as gr

iface = gr.Interface(launch, 
                     inputs=[
                         gr.Image(type='pil', label='Input Image'),
                        gr.Textbox(label='Question')
                     ], 
                     outputs=gr.Textbox(label='Answer'))

iface.launch(share=True, server_port=7831)
