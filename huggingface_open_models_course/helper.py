import io
import requests
import inflect
from PIL import Image, ImageDraw, ImageFont

def load_image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

def render_results_in_image(in_pil_img, in_results):
    # Use Pillow drawing instead of matplotlib so matplotlib is not required.
    img = in_pil_img.convert("RGBA").copy()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for prediction in in_results:
        x = int(prediction['box']['xmin'])
        y = int(prediction['box']['ymin'])
        w = int(prediction['box']['xmax'] - prediction['box']['xmin'])
        h = int(prediction['box']['ymax'] - prediction['box']['ymin'])

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline="green", width=3)

        # Prepare label and background
        label = f"{prediction['label']}: {round(prediction['score']*100, 1)}%"
        # Use textbbox to measure text size (compatible with modern Pillow versions)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position background; clamp so it doesn't go above the image
        pad = 4
        bg_y0 = max(0, y - text_height - pad)
        text_bg = [x, bg_y0, x + text_width + pad, bg_y0 + text_height + pad]
        draw.rectangle([int(c) for c in text_bg], fill="black")
        text_pos = (x + 2, bg_y0 + 2)
        draw.text((int(text_pos[0]), int(text_pos[1])), label, fill="red", font=font)

    return img

def summarize_predictions_natural_language(predictions):
    summary = {}
    p = inflect.engine()

    for prediction in predictions:
        label = prediction['label']
        if label in summary:
            summary[label] += 1
        else:
            summary[label] = 1

    result_string = "In this image, there are "
    for i, (label, count) in enumerate(summary.items()):
        count_string = p.number_to_words(count)
        result_string += f"{count_string} {label}"
        if count > 1:
          result_string += "s"

        result_string += " "

        if i == len(summary) - 2:
          result_string += "and "

    # Remove the trailing comma and space
    result_string = result_string.rstrip(', ') + "."

    return result_string


##### To ignore warnings #####
import warnings
import logging
from transformers import logging as hf_logging

def ignore_warnings():
    # Ignore specific Python warnings
    warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
    warnings.filterwarnings("ignore", message="Could not find image processor class")
    warnings.filterwarnings("ignore", message="The `max_size` parameter is deprecated")

    # Adjust logging for libraries using the logging module
    logging.basicConfig(level=logging.ERROR)
    hf_logging.set_verbosity_error()

########