# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython", "dotenv"]
# ///

# this example as shown in the course didn't work; what that means is, I was not able to use the huggingface
# inference API via REST to access the model; so instead, I just used the Transformers API directly


import os
import io
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper functions
import requests, json

#Image-to-text endpoint


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

def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_ITT_BASE']):
  headers = {
    "Authorization": f"Bearer {hf_api_key}",
    "Content-Type": "application/json"
  }
  data = { "inputs": inputs }
  if parameters is not None:
    data.update({"parameters": parameters})

  try:
    # print(f"Sending request to {ENDPOINT_URL} with inputs: {inputs}")
    response = requests.request("POST",
                  ENDPOINT_URL,
                  headers=headers,
                  data=json.dumps(data))
    # If server responded with non-2xx, print the raw body for debugging
    if not (200 <= response.status_code < 300):
      print(f"Status code: {response.status_code}")
      # Try to show text content (may be empty or non-json)
      try:
        body = response.text
      except Exception:
        body = repr(response.content)
      print(f"Response body: {body}")
      return None

    # For successful responses, attempt to parse JSON but guard against decode errors
    try:
      return json.loads(response.content.decode("utf-8"))
    except Exception as jde:
      print(f"Failed to decode JSON from response (status {response.status_code}): {jde}")
      # print raw content to help debugging
      try:
        print(f"Raw response content: {response.content!r}")
      except Exception:
        pass
      return None
  
  except Exception as e:
    print(f"Exception during request: {e}")
    return None

image_url = "https://free-images.com/sm/9596/dog_animal_greyhound_983023.jpg"


def image_to_base64_str(pil_image_or_path):
  """Accept either a PIL.Image.Image or a URL/local path string.

  Returns base64-encoded PNG string, or None on failure.
  """
  pil_image = None

  if isinstance(pil_image_or_path, str):
    # If a URL, download it; otherwise treat as a local path
    try:
      if pil_image_or_path.startswith("http://") or pil_image_or_path.startswith("https://"):
        resp = requests.get(pil_image_or_path)
        resp.raise_for_status()
        pil_image = Image.open(io.BytesIO(resp.content))
      else:
        pil_image = Image.open(pil_image_or_path)
    except Exception as e:
      print(f"Failed to open image from '{pil_image_or_path}': {e}")
      return None
  else:
    pil_image = pil_image_or_path

  if pil_image is None:
    print("No image available to convert to base64")
    return None

  try:
    # Ensure image is in a saveable mode
    if pil_image.mode in ("RGBA", "P", "LA"):
      converted = pil_image.convert("RGBA")
    else:
      converted = pil_image.convert("RGB")

    byte_arr = io.BytesIO()
    converted.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))
  except Exception as e:
    print(f"Failed to convert image to base64: {e}")
    return None


def captioner(image):
  base64_image = image_to_base64_str(image)
  if base64_image is None:
    print("Could not obtain base64 image for captioning")
    return None

  result = launch(base64_image)
  if not result:
    print("No result from get_completion")
    return None

  return result

import gradio as gr

if __name__ == '__main__':
    gr.close_all()
    demo = gr.Interface(fn=captioner,
                    inputs=[gr.Image(label="Upload image", type="pil")],
                    outputs=[gr.Textbox(label="Caption")],
                    title="Image Captioning with BLIP",
                    description="Caption any image using the BLIP model",
                    allow_flagging="never",
                    examples=["christmas_dog.jpeg", "bird_flight.jpeg", "cow.jpeg"])

    demo.launch(share=True, server_port=int(os.environ['PORT1']))