# /// script
# dependencies = ["openai", "torch", "dotenv", "ipython"]
# ///


import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
llama_api_key = os.environ['OPENROUTER_KEY']
llama_base_url = os.environ['OPENROUTER_BASE_URL']
llama_model = os.environ['OPENROUTER_MODEL']



from openai import OpenAI

def llama4(prompt,
    image_urls=[],
    model="Llama-4-Scout-17B-16E-Instruct-FP8",  # or Llama-4-Maverick-17B-128E-Instruct-FP8
    debug=False):
  image_urls_content = []
  for url in image_urls:
    image_urls_content.append(
        {"type": "image_url", "image_url": {"url": url}}) # TODO: for local image use {"url": "data:image/png;base64,..."}}

  content = [{"type": "text", "text": prompt}]
  content.extend(image_urls_content)

  client = OpenAI(api_key=llama_api_key, base_url=llama_base_url)

  response = client.chat.completions.create(
    model=llama_model,
    messages=[{
        "role": "user",
        "content": content
    }],
    temperature=0
  )

  if debug:
    print(response)

  return response.choices[0].message.content

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

def display_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

print(llama4("A brief history of AI in 3 short sentences."))