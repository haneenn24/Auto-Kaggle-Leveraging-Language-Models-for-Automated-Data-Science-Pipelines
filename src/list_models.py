import openai
import os

def list_available_models(api_key):
    openai.api_key = api_key
    try:
        models = openai.Model.list()
        for model in models['data']:
            print(model['id'])
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    list_available_models(api_key)
