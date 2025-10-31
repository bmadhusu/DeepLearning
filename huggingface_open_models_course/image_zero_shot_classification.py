# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython"]
# ///

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import CLIPModel

model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14")

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14")

from PIL import Image

image = Image.open("./kittens.jpeg")

image.show()

labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=labels,
                   images=image,
                   return_tensors="pt",
                   padding=True)

outputs = model(**inputs)
print(outputs.logits_per_image)

probs = outputs.logits_per_image.softmax(dim=1)[0]
print(probs)

probs = list(probs)
for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")

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

# iface.launch(share=True, server_port=7831)
