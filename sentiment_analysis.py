
import json
import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_completion(prompt:str, model="gpt-4-1106-preview", temperature:int=0)->json:
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
      ]
    )
    return response.choices[0].message.content


def sentiment_analyzer(input:str)->int:
    
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

    Comment: <{input}>
    """
    response = get_completion(prompt)
    try:
        res_dict = json.loads(response)
        print(res_dict)
        return res_dict['sentiment_score'], res_dict['offensive_score']
    except Exception:
        return -1


if __name__ == "__main__":
    demo = gr.Interface(fn=sentiment_analyzer,
                        inputs=gr.Textbox(label="Social Media Comment", lines=3.75), 
                        outputs=[gr.Textbox(label="Sentiment Score"), gr.Textbox(label="Offensive Language Score")],
                        title="Social Media Analysis",
                        description="""
                        <p>Enter a comment on a social platform, and our app will generate the corresponding sentiment and offensive language scores.
                        </p>

                        <table>
                        <thead>
                            <tr>
                            <th>Sentiment Analysis Scores</th>
                            <th>Offensive Language Scores</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                            <td>1 = Very Negative</td>
                            <td>1 = Not Offensive</td>
                            </tr>
                            <tr>
                            <td>2 = Negative</td>
                            <td>2 = Slightly Offensive</td>
                            </tr>
                            <tr>
                            <td>3 = Neutral</td>
                            <td>3 = Moderately Offensive</td>
                            </tr>
                            <tr>
                            <td>4 = Positive</td>
                            <td>4 = Offensive</td>
                            </tr>
                            <tr>
                            <td>5 = Very Positive</td>
                            <td>5 = Highly Offensive</td>
                            </tr>
                        </tbody>
                        </table>
                        """,
                        theme=gr.themes.Soft(),
                        css="footer {visibility: hidden}",
                        allow_flagging="never")
    demo.launch()