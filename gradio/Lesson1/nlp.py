# /// script
# dependencies = ["transformers", "torch", "gradio", "ipython", "dotenv"]
# ///

import os
import io
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE'] #NER endpoint


# Helper function
import requests, json, argparse

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    try:
        response = requests.request("POST",
                                    ENDPOINT_URL, headers=headers,
                                    data=json.dumps(data)
                                   )
        print(f"Status code: {response.status_code}")
        return json.loads(response.content.decode("utf-8"))
    except Exception as e:
        print(f"Exception during request: {e}")
        return None

text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
        as an 81-storey building, and the tallest structure in Paris. 
        Its base is square, measuring 125 metres (410 ft) on each side. 
        During its construction, the Eiffel Tower surpassed the Washington 
        Monument to become the tallest man-made structure in the world,
        a title it held for 41 years until the Chrysler Building
        in New York City was finished in 1930. It was the first structure 
        to reach a height of 300 metres. Due to the addition of a broadcasting 
        aerial at the top of the tower in 1957, it is now taller than the 
        Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
        Eiffel Tower is the second tallest free-standing structure in France 
        after the Millau Viaduct.''')

# print(f"{get_completion(text)}")

import gradio as gr

def summarize(input):
    output = get_completion(input)
    return output[0]['summary_text']



def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity_group'].startswith('I-') and merged_tokens[-1]['entity_group'].endswith(token['entity_group'][2:]):
            # If current token continues the entity of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}



def build_demo():
    """Create and return the Gradio Interface without launching it."""
    gr.close_all()
    demo = gr.Interface(fn=summarize,
                        inputs=[gr.Textbox(label="Text to summarize", lines=6)],
                        outputs=[gr.Textbox(label="Result", lines=3)],
                        title="Text summarization with distilbart-cnn",
                        description="Summarize any text using the `sshleifer/distilbart-cnn-12-6` model under the hood!"
                       )
    return demo

def build_ner_demo():
    gr.close_all()
    demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    flagging_mode="never",
                    examples=["My name is Andrew, I'm building DeeplearningAI and I live in California", "My name is Poli, I live in Vienna and work at HuggingFace"])
    return demo

def main():
    """CLI entrypoint. Use --text to summarize a string or run the Gradio demo.

    Flags:
      --text TEXT : summarize TEXT and print the result (no Gradio)
      --port PORT : port to use for Gradio (defaults to PORT1 env or 7860)
      --share      : pass share=True to Gradio.launch()
    """
    parser = argparse.ArgumentParser(description="Text Summary OR Named Entity Recognition")
    parser.add_argument("--summary", action="store_true", help="Run text summarization demo")
    parser.add_argument("--port", type=int, default=int(os.environ.get('PORT1', 7860)), help="Server port for Gradio")
    parser.add_argument("--share", action="store_true", help="Create a public share link for Gradio")

    args = parser.parse_args()

    if args.summary:
        demo_summary = build_demo()
        demo_summary.launch(share=args.share, server_port=args.port)
    else:
        print("Launching NER demo...")
        demo_ner = build_ner_demo()
        demo_ner.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()



# [{'entity_group': 'PER', 'score': 0.9990625, 'word': 'Andrew', 'start': 11, 'end': 17}, {'entity_group': 'ORG', 'score': 0.89605016, 'word': 'DeepLearningAI', 'start': 32, 'end': 46}, {'entity_group': 'LOC', 'score': 0.99969244, 'word': 'California', 'start': 61, 'end': 71}]