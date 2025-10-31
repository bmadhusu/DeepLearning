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

# Helper function
import requests, json

#Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_BASE']):
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }   
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    try:
        response = requests.request("POST",
                                    ENDPOINT_URL,
                                    headers=headers,
                                    data=json.dumps(data))
    except Exception as e:
        print(f"Exception during request: {e}")
        return None

    print(f"Status code: {response.status_code}")

    # Inspect Content-Type to decide how to decode the body
    content_type = response.headers.get('Content-Type', '')
    # If the server returned JSON, parse it
    if 'json' in content_type.lower() or response.status_code == 204:
        try:
            return response.json()
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
            # fallthrough to try text decode

    # If the server returned an image (binary), return base64 string
    if content_type.startswith('image/') or 'application/octet-stream' in content_type.lower():
        try:
            b64 = base64.b64encode(response.content).decode('ascii')
            return {'image_base64': b64, 'content_type': content_type}
        except Exception as e:
            print(f"Failed to encode image bytes: {e}")
            return None

    # As a last resort, try to decode text safely
    try:
        text = response.content.decode('utf-8')
        # Try json.loads on text
        try:
            return json.loads(text)
        except Exception:
            return text
    except UnicodeDecodeError as ude:
        print(f"UnicodeDecodeError while decoding response: {ude}")
        # Return raw bytes base64-encoded so caller can inspect
        try:
            return {'raw_base64': base64.b64encode(response.content).decode('ascii'), 'content_type': content_type}
        except Exception as e:
            print(f"Failed to base64-encode raw content: {e}")
            return None

prompt = "a dog in a park"
result = get_completion(prompt)
print(f"keys in result: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")

def _save_base64_image(b64_str, filename="generated_image.png"):
    try:
        data = base64.b64decode(b64_str)
        with open(filename, "wb") as f:
            f.write(data)
        return filename
    except Exception as e:
        print(f"Failed to save image to {filename}: {e}")
        return None


def _handle_result(res):
    print(f"In _handle_result with res type: {type(res)}")
    if res is None:
        print("No result from get_completion")
        return

    # If the API returned a dict with an image
    if isinstance(res, dict):
        if res.get('image_base64'):
            fname = _save_base64_image(res['image_base64'])
            if fname:
                print(f"Saved generated image to: {fname}")
                try:
                    IPython.display.display(IPython.display.Image(filename=fname))
                except Exception:
                    pass
            return

        if res.get('raw_base64'):
            fname = _save_base64_image(res['raw_base64'], filename="generated_raw.png")
            if fname:
                print(f"Saved raw binary to: {fname}")
                try:
                    IPython.display.display(IPython.display.Image(filename=fname))
                except Exception:
                    pass
            return

        # Otherwise, just print the dict
        print("Received dict result:", res)
        return

    # If string, print it (could be text or JSON string)
    if isinstance(res, str):
        print(res)
        return

    # Fallback: print the raw result
    print("Result:", res)


import gradio as gr 

#A helper function to convert the PIL image to base64
#so you can send it to the API
import gradio as gr 

#A helper function to convert the PIL image to base64 
# so you can send it to the API
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def generate(prompt, negative_prompt, steps, guidance, width, height):
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(prompt, params)
    print(f"keys in output: {list(output.keys()) if isinstance(output, dict) else 'not a dict'}")
    pil_image = base64_to_pil(output['image_base64'])
    return pil_image

import gradio as gr

# demo = gr.Interface(fn=generate,
#                     inputs=[
#                         gr.Textbox(label="Your prompt"),
#                         gr.Textbox(label="Negative prompt"),
#                         gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
#                                  info="In how many steps will the denoiser denoise the image?"),
#                         gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7, 
#                                   info="Controls how much the text prompt influences the result"),
#                         gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512),
#                         gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512),
#                     ],
#                     outputs=[gr.Image(label="Result")],
#                     title="Image Generation with Stable Diffusion",
#                     description="Generate any image with Stable Diffusion",
#                     flagging_mode="never"
#                     )

# with gr.Blocks() as demo:
#     gr.Markdown("# Image Generation with Stable Diffusion")
#     prompt = gr.Textbox(label="Your prompt")
#     with gr.Row():
#         with gr.Column():
#             negative_prompt = gr.Textbox(label="Negative prompt")
#             steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
#                       info="In many steps will the denoiser denoise the image?")
#             guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
#                       info="Controls how much the text prompt influences the result")
#             width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
#             height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
#             btn = gr.Button("Submit")
#         with gr.Column():
#             output = gr.Image(label="Result")

#     btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])

with gr.Blocks() as demo:
    gr.Markdown("# Image Generation with Stable Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt") #Give prompt some real estate
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit") #Submit button side by side!
    with gr.Accordion("Advanced options", open=False): #Let's hide the advanced options!
            negative_prompt = gr.Textbox(label="Negative prompt")
            with gr.Row():
                with gr.Column():
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=25,
                      info="In many steps will the denoiser denoise the image?")
                    guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=7,
                      info="Controls how much the text prompt influences the result")
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=512, step=64, value=512)
                    height = gr.Slider(label="Height", minimum=64, maximum=512, step=64, value=512)
    output = gr.Image(label="Result") #Move the output up too
            
    btn.click(fn=generate, inputs=[prompt,negative_prompt,steps,guidance,width,height], outputs=[output])



if __name__ == '__main__':
    gr.close_all()
    demo.launch(server_port=int(os.environ['PORT1']))
