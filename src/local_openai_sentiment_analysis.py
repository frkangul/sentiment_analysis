
import json
import logging
import sqlite3
from contextlib import contextmanager

import gradio as gr
from utils import (
    LOCAL_MODEL,
    OPENAI_MODEL,
    get_completion_local,
    get_completion_openai,
    gr_description,
    nllb_translate_tr_to_eng,
)

# Define constants
ERROR_CODE = -1

@contextmanager
def get_db_connection():
    con = sqlite3.connect("../logs.db", check_same_thread=False)
    try:
        cur = con.cursor()
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS logs(
                        ID INTEGER PRIMARY KEY, 
                        input TEXT, 
                        model TEXT,
                        eng_input TEXT,
                        sentiment_score INT,
                        offensive_score INT,
                        RESPONSE TEXT,
                        ERROR TEXT,
                        timestamp DATE DEFAULT (datetime('now','localtime'))
                    )
                """)
        yield con
    finally:
        con.close()

def sentiment_analyzer(input:str, is_local:bool)->int:
    """
    Generate sentiment and offensive lang analyze

    Args:
        input (str): social media comment in turkish

    Returns:
        response['sentiment_score'] (int): sentiment score: 1, 2, 3, 4, 5
        response['offensive_score'] (int): offensive lang score: 1, 2, 3, 4, 5
    """

    logger.info(f"Original Input: {input}")
    if is_local:
        input_eng = nllb_translate_tr_to_eng(article=input)
        logger.info(f"Translated Input: {input_eng}")
        comment = input_eng
        MODEL = LOCAL_MODEL
        get_completion = get_completion_local
    else:
        input_eng = None
        comment = input
        MODEL = OPENAI_MODEL
        get_completion = get_completion_openai
    logger.info(f"Model: {MODEL}")
    
    prompt = f"""
    Your task is to perform the following actions based on a social media comment, delimited by <>:
    
    1 - Assign a sentiment score from 1 to 5 for the comment, where: \
        1 = Very Negative
        2 = Negative
        3 = Neutral
        4 = Positive
        5 = Very Positive
    2 - Assign an offensive language score from 1 to 5 for the comment, where:
        1 = Not Offensive
        2 = Slightly Offensive
        3 = Moderately Offensive
        4 = Offensive
        5 = Highly Offensive
    
    Format your response as a JSON object with the keys \
    'sentiment_score' and 'offensive_score'. 
    Make your response as short as possible without any additional explanation.

    Comment: <{comment}>
    """

    response = get_completion(prompt)
    logger.info(f"Raw Response: {response}")
    with get_db_connection() as con:
        cur = con.cursor()
        try:
            # Decode Unicode escape sequences
            response_decoded = response.encode('utf-8').decode('unicode_escape')
            res_dict = json.loads(response_decoded)

            # WRITE INTO DB
            cur.execute("""
                INSERT INTO logs(ID, input, model, eng_input, sentiment_score, offensive_score) VALUES
                    (NULL, ?, ?, ?, ?, ?)
            """, (input, MODEL, input_eng, res_dict['sentiment_score'], res_dict['offensive_score']))
            con.commit()
            return res_dict['sentiment_score'], res_dict['offensive_score']
        except Exception as e:
            logger.error(e)
            return ERROR_CODE, ERROR_CODE

if __name__ == "__main__":
    # Define logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename='../chatgpt_pipeline.log',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    demo = gr.Interface(fn=sentiment_analyzer,
                        inputs=[gr.Textbox(label="Social Media Comment", lines=1.8), gr.Checkbox(label="Local LLM")], 
                        outputs=[gr.Textbox(label="Sentiment Score"), gr.Textbox(label="Offensive Language Score")],
                        title="Social Media Analysis",
                        description=gr_description,
                        theme=gr.themes.Soft(),
                        css="footer {visibility: hidden}",
                        allow_flagging="never")
    demo.launch(share=True)