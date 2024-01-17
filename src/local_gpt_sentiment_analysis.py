
import json
import sqlite3

import gradio as gr
from utils import (
    LOCAL_MODEL,
    OPENAI_MODEL,
    get_completion_local,
    get_completion_openai,
    nllb_translate_tr_to_eng,
)


def sentiment_analyzer(input:str, is_local:bool)->int:
    """
    Generate sentiment and offensive lang analyze

    Args:
        input (str): social media comment in turkish

    Returns:
        response['sentiment_score'] (int): sentiment score: 1, 2, 3, 4, 5
        response['offensive_score'] (int): offensive lang score: 1, 2, 3, 4, 5
    """

    print(f"Original Input: {input}")
    if is_local:
        input_eng = nllb_translate_tr_to_eng(article=input)
        print(f"Translated Input: {input_eng}")
        comment = input_eng
        get_completion = get_completion_local
    else:
        comment = input
        get_completion = get_completion_openai

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

    Comment: <{comment}>
    """

    response = get_completion(prompt)
    print(response)
    try:
        res_dict = json.loads(response)
        print(50*"-")
        # WRITE INTO DB
        if is_local:
            cur.execute("""
                INSERT INTO logs(ID, input, model, eng_input, sentiment_score, offensive_score) VALUES
                    (NULL, ?, ?, ?, ?, ?)
            """, (input, LOCAL_MODEL, input_eng, res_dict['sentiment_score'], res_dict['offensive_score']))
        else:
            cur.execute("""
                INSERT INTO logs(ID, input, model, sentiment_score, offensive_score) VALUES
                    (NULL, ?, ?, ?, ?)
            """, (input, OPENAI_MODEL, res_dict['sentiment_score'], res_dict['offensive_score']))

        con.commit()

        return res_dict['sentiment_score'], res_dict['offensive_score']
    except Exception as e:
        print(e)

if __name__ == "__main__":
    con = sqlite3.connect("log.db", check_same_thread=False)
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS logs(ID INTEGER PRIMARY KEY, input TEXT, model TEXT, eng_input TEXT, sentiment_score INT, offensive_score INT, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
    
    demo = gr.Interface(fn=sentiment_analyzer,
                        inputs=[gr.Textbox(label="Social Media Comment", lines=1.8), gr.Checkbox(label="Local LLM")], 
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
    demo.launch() # share=True