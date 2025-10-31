# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython", "dotenv"]
# ///


import os
import io
import IPython.display
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

#### Helper function
import requests, json

from transformers import BlipForConditionalGeneration

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")

def launch(pil_image):

    inputs = processor(pil_image,return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

#Here we are going to call multiple endpoints!
def get_completion(inputs, parameters=None, ENDPOINT_URL=""):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL,
                                headers=headers,
                                data=json.dumps(data))
    
    b64 = base64.b64encode(response.content).decode('ascii')
    return {'image_base64': b64, 'content_type': response.headers.get('Content-Type', '')}
    
    # return json.loads(response.content.decode("utf-8"))

#text-to-image
TTI_ENDPOINT = os.environ['HF_API_TTI_BASE']
#image-to-text
ITT_ENDPOINT = os.environ['HF_API_ITT_BASE']

#Bringing the functions from lessons 3 and 4!
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def captioner(image):
    base64_image = image_to_base64_str(image)
    # result = get_completion(base64_image, None, ITT_ENDPOINT)
    result = launch(base64_image)
    
    return result



def generate(prompt):
    output = get_completion(prompt, None, TTI_ENDPOINT)
    result_image = base64_to_pil(output['image_base64'])
    return result_image

import gradio as gr

# with gr.Blocks() as demo:
#     gr.Markdown("# Describe-and-Generate game 🖍️")
#     image_upload = gr.Image(label="Your first image",type="pil")
#     btn_caption = gr.Button("Generate caption")
#     caption = gr.Textbox(label="Generated caption")
    
#     btn_caption.click(fn=captioner, inputs=[image_upload], outputs=[caption])

# with gr.Blocks() as demo:
#     gr.Markdown("# Describe-and-Generate game 🖍️")
#     image_upload = gr.Image(label="Your first image",type="pil")
#     btn_caption = gr.Button("Generate caption")
#     caption = gr.Textbox(label="Generated caption")
#     btn_image = gr.Button("Generate image")
#     image_output = gr.Image(label="Generated Image")
#     btn_caption.click(fn=captioner, inputs=[image_upload], outputs=[caption])
#     btn_image.click(fn=generate, inputs=[caption], outputs=[image_output])

def caption_and_generate(image):
    caption = captioner(image)
    image = generate(caption)
    return [caption, image]

with gr.Blocks() as demo:
    gr.Markdown("# Describe-and-Generate game 🖍️")
    image_upload = gr.Image(label="Your first image",type="pil")
    btn_all = gr.Button("Caption and generate")
    caption = gr.Textbox(label="Generated caption")
    image_output = gr.Image(label="Generated Image")

    btn_all.click(fn=caption_and_generate, inputs=[image_upload], outputs=[caption, image_output])


gr.close_all()
demo.launch(server_port=int(os.environ['PORT1']))