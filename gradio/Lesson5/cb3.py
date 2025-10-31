# /// script
# dependencies = ["openai", "transformers", "torch", "gradio", "ipython", "dotenv"]
# ///

import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ['OPENROUTER_KEY'],
)

def gen_ai(input, **kwargs):
    print("Generating with kwargs:", kwargs)
    response = client.chat.completions.create(
      model="openai/gpt-4o",
      messages=[
        {
          "role": "user",
          "content": input
        }
      ],
      **kwargs
    )
    return response.choices[0].message.content

def gen_ai_stream(input, **kwargs):
    print("Generating with kwargs:", kwargs)
    response = client.chat.completions.create(
      model="openai/gpt-4o",
      messages=[
        {
          "role": "user",
          "content": input
        }
      ],
      stream=True,
      **kwargs
    )
    return response


#Back to Lesson 2, time flies!
import gradio as gr

def generate(input, slider):
    output = gen_ai(input, max_tokens=slider)
    return output

import random

def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = gen_ai_stream(prompt,
                            max_tokens=1024,
                            stop=["\nUser:", "<|endoftext|>"],
                            temperature=temperature)
                            #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, chunk in enumerate(stream):
            text_token = ""
            if chunk.choices and chunk.choices[0].delta.content:
                text_token = chunk.choices[0].delta.content
           
            # check for completion
            if chunk.choices and chunk.choices[0].finish_reason:
                break

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            if not text_token:
                # skip empty tokens
                continue

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion(label="Advanced options",open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.queue().launch(server_port=int(os.environ['PORT1']))    