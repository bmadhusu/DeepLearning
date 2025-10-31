# /// script
# dependencies = ["transformers", "timm", "inflect", "phonemizer", "gradio", "ipython"]
# ///

from transformers.utils import logging

logging.set_verbosity_error()

from transformers import pipeline

from pathlib import Path

import shutil
import sys

# Ensure espeak (or espeak-ng) is installed for phonemizer
if shutil.which("espeak") is None and shutil.which("espeak-ng") is None:
    raise RuntimeError(
        "espeak (or espeak-ng) not found. Install it and re-run.\n"
        "macOS (Homebrew):    brew install espeak\n"
        "Ubuntu/Debian:       sudo apt-get update && sudo apt-get install -y espeak\n"
        "Fedora:              sudo dnf install -y espeak\n"
        "If you have espeak-ng only, symlink it so it's named 'espeak' in PATH:\n"
        "  sudo ln -s \"$(which espeak-ng)\" /usr/local/bin/espeak"
    )



narrator = pipeline(
    "text-to-speech",
    model="kakao-enterprise/vits-ljs"
)

text = """
You gotta be joking punk ass mofo, \
HuggingFace, Microsoft, the University of Washington, \
Carnegie Mellon University, and the Hebrew University of \
Jerusalem developed a tool that measures atmospheric \
carbon emitted by cloud servers while training machine \
learning models. After a model’s size, the biggest variables \
were the server’s location and time of day it was active.
"""

narrated_text = narrator(text)

