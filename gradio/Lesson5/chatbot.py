# /// script
# dependencies = ["httpx", "text_generation", "transformers", "torch", "gradio", "ipython", "dotenv"]
# ///


import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60
import logging

# Enable debug logging for httpx
import sys
import httpx
import inspect

# Store original methods
original_request = requests.Session.request

def logging_request(self, method, url, **kwargs):
    print("\n" + "="*80)
    print(f"REQUEST: {method} {url}")
    print(f"Headers: {kwargs.get('headers', {})}")
    
    # Print request body
    if 'json' in kwargs:
        print(f"JSON Body: {kwargs['json']}")
    elif 'data' in kwargs:
        print(f"Data Body: {kwargs['data']}")
    print("="*80 + "\n")
    
    # Make the actual request
    response = original_request(self, method, url, **kwargs)
    
    print("\n" + "="*80)
    print(f"RESPONSE: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    try:
        print(f"Body: {response.text[:2000]}")  # First 2000 chars
    except:
        print("Could not decode response body")
    print("="*80 + "\n")
    
    return response

# Apply monkey patch
requests.Session.request = logging_request

# Enable all HTTP-related logging
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("httpcore").setLevel(logging.DEBUG)


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

prompt = "Has math been invented or discovered?"

# Test the endpoint directly
response = requests.post(
    os.environ['HF_API_FALCOM_BASE'],
    headers={"Authorization": f"Bearer {hf_api_key}"},
    json={"inputs": prompt, "parameters": {"max_new_tokens": 256}},
    timeout=120
)

print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")
print(f"Response Headers: {response.headers}")

#FalcomLM-instruct endpoint on the text_generation library
# client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Bearer {hf_api_key}"}, timeout=120)

print(client.generate(prompt, max_new_tokens=256))
    #   .generated_text)



#Back to Lesson 2, time flies!
import gradio as gr
def generate(input, slider):
    output = client.generate(input, max_new_tokens=slider).generated_text
    return output

demo = gr.Interface(fn=generate, 
                    inputs=[gr.Textbox(label="Prompt"), 
                            gr.Slider(label="Max new tokens", 
                                      value=20,  
                                      maximum=1024, 
                                      minimum=1)], 
                    outputs=[gr.Textbox(label="Completion")])

gr.close_all()
demo.launch(server_port=int(os.environ['PORT1']))