# /// script
# dependencies = ["huggingface_hub", "transformers", "torch", "gradio", "ipython", "dotenv"]
# ///


import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from huggingface_hub import InferenceClient

client = InferenceClient(
    model=os.environ['HF_API_FALCOM_BASE'],
    api_key=hf_api_key,
)

messages = [
    {
        "role": "user",
        "content": "Has math been invented or discovered?",
    }
]

print(client.chat_completion(messages, max_tokens=256))