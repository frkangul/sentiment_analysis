import json

import gradio as gr
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tur_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

OLLAMA_MODEL = "mistral"
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

def get_completion(prompt:str, model:str=OLLAMA_MODEL, url:str="http://localhost:11434/api/generate")->json:
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

def sentiment_analyzer(input:str)->int:
    """
    Generate sentiment and offensive lang analyze

    Args:
        input (str): social media comment in turkish

    Returns:
        response['sentiment_score'] (int): sentiment score: 1, 2, 3, 4, 5
        response['offensive_score'] (int): offensive lang score: 1, 2, 3, 4, 5
    """

    input_eng = nllb_translate_tr_to_eng(article=input)
    print(f"Original Input: {input}")
    print(f"Translated Input: {input_eng}")
    
    prompt = f"""
    Your task is to perform the following actions based on the social media comment, delimited by <>:
    
    1 - Generate the sentiment analysis for the comment, \
        assign a score from 1 to 5, where:
        1 = Very Negative
        2 = Negative
        3 = Neutral
        4 = Positive
        5 = Very Positive
    2 - Generate the offensive language detection for the comment, \
        assign a score from 1 to 5, where:
        1 = Not Offensive
        2 = Slightly Offensive
        3 = Moderately Offensive
        4 = Offensive
        5 = Highly Offensive
    
    Format your response as a JSON object with the keys \
    'sentiment_score' and 'offensive_score'. 

    Comment: <{input_eng}>
    """

    response = get_completion(prompt)
    print(response)
    try:
        res_dict = json.loads(response)
        print(50*"-")
        return res_dict['sentiment_score'], res_dict['offensive_score']
    except Exception:
        return -1

if __name__ == "__main__":
    demo = gr.Interface(fn=sentiment_analyzer,
                        inputs=gr.Textbox(label="Social Media Comment", lines=3.75), 
                        outputs=[gr.Textbox(label="Sentiment Score"), gr.Textbox(label="Offensive Language Score")],
                        title="Social Media Analysis",
                        description="""
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
                        """,
                        theme=gr.themes.Soft(),
                        css="footer {visibility: hidden}",
                        allow_flagging="never")
    demo.launch(share=True)