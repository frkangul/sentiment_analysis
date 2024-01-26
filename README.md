# Turkish LLM: Sentiment and Offensive Language Analysis

This repository contains a sentiment and offensive language analysis tool that uses both local and OpenAI models to analyze social media comments. The tool translates Turkish comments into English, then generates sentiment and offensive language scores.

## Features

- Translation from Turkish to English using Facebook's NLLB-200-Distilled-600M model.
- Sentiment and offensive language analysis using a local model or OpenAI's GPT-3 model.
- User-friendly interface using Gradio.
- Logging of analysis results in a SQLite database.

## Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/sentiment_analysis.git
cd sentiment_analysis
poetry install
```

## Usage

Run the main script:
```bash
python src/local_openai_sentiment_analysis.py
```

This will launch a Gradio interface in your web browser where you can input a social media comment and choose whether to use the local or OpenAI model for sentiment and offensive language analysis.

## Configuration

After installing OLLAMA on your local server, you can configure the OLLAMA model, OpenAI model, API key, URL for OLLAMA servers using environment variables. Create a `.env` file in the root directory of the project with the following variables:
```bash
LOCAL_MODEL="mistral"
OPENAI_MODEL="gpt-3.5-turbo"
OPENAI_API_KEY=your_openai_api_key
URL="http://localhost:11434"
```

Replace your_openai_api_key with your actual OpenAI API key, gpt-3.5-turbo with the OpenAI model you want to use, and mistral with the OLLAMA model you want to use.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.