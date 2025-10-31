# /// script
# dependencies = ["torch", "transformers", "datasets<3.2.0", "soundfile", "librosa", "gradio", "ipython"]
# ///

import os
# Disable torchcodec and use soundfile for audio decoding
os.environ["HF_DATASETS_DISABLE_TORCHCODEC"] = "1"

import transformers
import datasets
import soundfile
import librosa
import gradio as gr

from transformers.utils import logging
logging.set_verbosity_error()

from datasets import load_dataset

dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)

example = next(iter(dataset))



# Create a Gradio interface to play the audio
def get_audio_example():
    return (example["audio"]["sampling_rate"], example["audio"]["array"]), example["text"]

# with gr.Blocks() as demo:
#     gr.Markdown("# LibriSpeech Audio Example")
#     audio_output = gr.Audio(label="Audio Sample", type="numpy")
#     text_output = gr.Textbox(label="Transcription")

#     # Load the example immediately
#     demo.load(get_audio_example, outputs=[audio_output, text_output])

# demo.launch()

from transformers import pipeline
asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en")

print(f"The sampling rate in model is: {asr.feature_extractor.sampling_rate}")
print("Example data:")

print(f"Audio sampling rate: {example['audio']['sampling_rate']}")
print(f"Audio array shape: {example['audio']['array'].shape}")

result = asr({"array": example["audio"]["array"], "sampling_rate": example["audio"]["sampling_rate"]})
print(f"Transcription from model is: {result}")
print(f"Ground truth text: {example['text']}")


def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    print(f"filepath: {filepath}, type: {type(filepath)}")
    output = asr(filepath)
    return output["text"]

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never")

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    flagging_mode="never",
)

with gr.Blocks() as demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )

demo.launch()