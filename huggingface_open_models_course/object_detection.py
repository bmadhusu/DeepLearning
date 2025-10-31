# /// script
# dependencies = ["transformers", "timm", "inflect", "phonemizer", "gradio", "ipython"]
# ///


from helper import load_image_from_url, render_results_in_image
import gradio as gr


from transformers import pipeline

from transformers.utils import logging
logging.set_verbosity_error()

from helper import ignore_warnings
ignore_warnings()

od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

from PIL import Image

raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((569, 491))

pipeline_output = od_pipe(raw_image)

processed_image = render_results_in_image(
    raw_image, 
    pipeline_output)


# def show_image():
#     return processed_image

# image_viewer = gr.Interface(
#     fn=show_image,
#     inputs=None,
#     outputs=gr.Image()
# )

# with gr.Blocks() as demo:
#     gr.TabbedInterface(
#         [image_viewer],
#         ["View Image"],
#     )

def get_pipeline_prediction(pil_image):
    
    pipeline_output = od_pipe(pil_image)
    
    processed_image = render_results_in_image(pil_image,
                                            pipeline_output)
    return processed_image

demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image", 
                  type="pil"),
  outputs=gr.Image(label="Output image with predicted instances",
                   type="pil")
)

demo.launch()

