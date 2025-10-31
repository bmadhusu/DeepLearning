# /// script
# dependencies = ["transformers", "timm", "gradio", "torch", "torchvision", "ipython"]
# ///

import gradio as gr
import torch
import numpy as np

# Ensure PyTorch uses float32 by default. MPS backend does not support float64 tensors
# and the SAM pipeline may create double tensors; set the default dtype before creating
# the pipeline so tensors are float32 when moved to the MPS device.
torch.set_default_dtype(torch.float32)

from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline

# Run the pipeline on CPU to avoid MPS float64 incompatibility on Apple Silicon.
# If you want to use MPS/GPU, you can try converting inputs to float32 before calling the pipeline
# or set up a different device handling strategy.
sam_pipe = pipeline("mask-generation",
    "Zigeng/SlimSAM-uniform-77",
    device=-1)

from PIL import Image

raw_image = Image.open('meta_llamas.jpg')
raw_image.resize((720, 375))

# commenting out below to save time
# output = sam_pipe(raw_image, points_per_batch=32)

from helper2 import show_pipe_masks_on_image


def get_processed_image(_=None):
    """Gradio will call this function and return the processed PIL image.
    The input argument is ignored because we already computed `raw_image` and `output`.
    """
    return show_pipe_masks_on_image(raw_image, output)


# demo = gr.Interface(
#   fn=get_processed_image,
#   inputs=gr.Image(label="Input image", 
#                   type="pil"),
#   outputs=gr.Image(label="Output image with predicted instances",
#                    type="pil")
# )

# demo.launch()

from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained(
    "Zigeng/SlimSAM-uniform-77")

processor = SamProcessor.from_pretrained(
    "Zigeng/SlimSAM-uniform-77")

input_points = [[[1600, 700]]]

inputs = processor(raw_image,
                 input_points=input_points,
                 return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predicted_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)
print(len(predicted_masks))

predicted_mask = predicted_masks[0]
print(predicted_mask.shape)

print(outputs.iou_scores)

from helper2 import show_mask_on_image

# Uncomment below to visualize individual masks

# for i in range(3):
#     show_mask_on_image(raw_image, predicted_mask[:, i])

##### Below is using a depth estimation model #####

depth_estimator = pipeline(task="depth-estimation",
                        model="Intel/dpt-hybrid-midas")

raw_image = Image.open('gradio_tamagochi_vienna.png')
raw_image.resize((806, 621))

def launch(input_image):
    output = depth_estimator(input_image)

    # Prepare depth tensor for interpolation: ensure it's a torch tensor with shape (N, C, H, W)
    depth_tensor = output.get("predicted_depth")
    if depth_tensor is None:
        raise KeyError("depth_estimator output missing 'predicted_depth'")

    # Convert numpy -> torch if needed
    if isinstance(depth_tensor, np.ndarray):
        depth_tensor = torch.from_numpy(depth_tensor)

    # Ensure float32
    depth_tensor = depth_tensor.to(dtype=torch.float32)

    # Normalize dims to (N, C, H, W)
    if depth_tensor.ndim == 2:  # (H, W)
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
    elif depth_tensor.ndim == 3:
        # Could be (N, H, W) or (C, H, W). Handle common case (N, H, W) -> add channel dim
        if depth_tensor.shape[0] == 1 or depth_tensor.shape[0] == depth_tensor.shape[0]:
            depth_tensor = depth_tensor.unsqueeze(1)
        else:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(1)
    elif depth_tensor.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected depth tensor shape: {depth_tensor.shape}")

    # target size expects (H, W)
    target_size = (raw_image.size[1], raw_image.size[0])

    prediction = torch.nn.functional.interpolate(
        depth_tensor,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )

    print(prediction.shape)
    print(raw_image.size[::-1],)

    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

# below line opens image in Preview app
# launch(raw_image).show()

iface = gr.Interface(launch, 
                     inputs=gr.Image(type='pil'), 
                     outputs=gr.Image(type='pil'))

iface.launch(share=True, server_port=7832)