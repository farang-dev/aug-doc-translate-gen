# AI Translation App

A Streamlit-based translation application that uses AI to translate text while maintaining the style and tone of reference documents.

## Features

- Translation between multiple languages
- Reference document upload for style matching
- Chat-based interface with conversation history
- Custom conversation titles
- Document management

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.streamlit/secrets.toml` file based on the `.streamlit/secrets.toml.example` template
4. Run the app: `streamlit run app.py`

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `SUPABASE_URL`: Your Supabase URL
- `SUPABASE_KEY`: Your Supabase key
- `STREAMLIT_ENV`: Set to "development" for local development or "production" for deployment

## Deployment

This app is designed to be deployed on Streamlit Community Cloud.
