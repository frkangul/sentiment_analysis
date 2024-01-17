import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tur_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

LOCAL_MODEL = "mistral"
OPENAI_MODEL = "gpt-4-1106-preview"

def nllb_translate_tr_to_eng(article:str = "Bugün hava güneşli ama benim havam bulutlu"):
    """Translate from turkish to english using facebook:nllb-200-distilled-600M on hface. 
    For default article, 
    - it takes 17.3s
    - after removing imports outside, it takes 10.4s
    - after removing imports, tokenizer, model outside, it takes 1.7s

    Args:
        article (str, optional): turkish input. Defaults to "Bugün hava güneşli ama benim havam bulutlu".

    Returns:
        eng (str): english output.
    """
    inputs = tokenizer(article, return_tensors="pt") # Return PyTorch torch.Tensor objects
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=30)
    eng = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return eng

def mbart_translate_tr_to_eng(article:str = "Bugün hava güneşli ama benim havam bulutlu"):
    """Translate from turkish to english using facebook:mbart-large-50-many-to-many-mmt on hface. 
    For default article, 
    - it takes 24.1s
    - after removing imports outside, it takes 12.5s
    - after removing imports, tokenizer, model outside, it takes 1.9s

    Args:
        article (str, optional): turkish input. Defaults to "Bugün hava güneşli ama benim havam bulutlu".

    Returns:
        eng (str): english output.
    """
    from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")   
    tokenizer.src_lang = "tr_TR"

    inputs = tokenizer(article, return_tensors="pt")
    generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    eng = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return eng

def get_completion_local(prompt:str, model:str=LOCAL_MODEL, url:str="http://localhost:11434/api/generate")->json:
    """
    Send a single prompt to local ollama API and return the response.
    See https://github.com/jmorganca/ollama.

    Args:
        prompt (str): prompt to send to the local API
        model (str, optional): Ollama model type. Defaults to "phi".
        url (str, optional): Ollama REST API URL

    Returns:
        response_content (str): response from local ollama API
    """
    data = {
        "prompt": prompt, "model": model, "stream": True
    }
    response = requests.post(url, json=data)
    response.raise_for_status()
    parts = []
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])

        content = body.get("response", "")
        parts.append(content)

        if body.get("done"):
            break

    response_content = "".join(parts).strip()
    return response_content

def get_completion_openai(prompt:str, model:str=OPENAI_MODEL, temperature:int=0)->json:
    """
    Send a single prompt to the OpenAI API and return the response.

    Args:
        prompt (str): prompt to send to the API
        model (str, optional): OpenAI model type. Defaults to "gpt-3.5-turbo".
        temperature (int, optional): degree of randomness of the model's output. It changes the variety of model's response. Defaults to 0.

    Returns:
        json: response from the OpenAI API
    """

    response = client.chat.completions.create(
      model=model,
      response_format={ "type": "json_object" },
      messages=[
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
      ],
      temperature=temperature,
    )
    return response.choices[0].message.content

gr_description = """
                <!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    /* Add some basic styling to the tables */
                    table {
                    border-collapse: collapse;
                    width: 50%;
                    margin-bottom: 20px;
                    }

                    th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                    }

                    th {
                    background-color: #f2f2f2;
                    }
                </style>
                </head>
                <body>

                <p>Enter a comment on a social platform, and our app will generate the corresponding sentiment score and offensive language score.</p>

                <!-- Use details and summary for toggle functionality -->
                <details>
                <summary>Score Explanation</summary>

                <!-- Sentiment Analysis Scores Table -->
                <table>
                    <thead>
                    <tr>
                        <th>Sentiment Analysis Scores</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr><td>1 = Very Negative</td></tr>
                    <tr><td>2 = Negative</td></tr>
                    <tr><td>3 = Neutral</td></tr>
                    <tr><td>4 = Positive</td></tr>
                    <tr><td>5 = Very Positive</td></tr>
                    </tbody>
                </table>

                <!-- Offensive Language Scores Table -->
                <table>
                    <thead>
                    <tr>
                        <th>Offensive Language Scores</th>
                    </tr>
                    </thead>
                    <tbody>
                    <tr><td>1 = Not Offensive</td></tr>
                    <tr><td>2 = Slightly Offensive</td></tr>
                    <tr><td>3 = Moderately Offensive</td></tr>
                    <tr><td>4 = Offensive</td></tr>
                    <tr><td>5 = Highly Offensive</td></tr>
                    </tbody>
                </table>
                </details>

                </body>
                </html>
                """