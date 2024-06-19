import sys
import os
from unittest.mock import patch
import time

# Add src to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config_loader import load_config
from llm_gpt import GPTModel
from logging_config import setup_logging
import logging

def main():
    setup_logging()
    
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    llm = GPTModel(api_key)
    
    # Process each dataset
    for dataset_name, dataset_config in config['datasets'].items():
        logging.info(f"Processing dataset: {dataset_name}")
        
        if 'train_path' in dataset_config and 'test_path' in dataset_config:
            dataset_paths = (dataset_config['train_path'], dataset_config['test_path'])
        else:
            dataset_paths = dataset_config['path']
        
        try:
            # Generate code
            generated_code = llm.generate_code(dataset_paths)
            logging.info(f"Generated code for {dataset_name}:\n{generated_code}")
            
        except Exception as e:
            if "insufficient_quota" in str(e):
                logging.error(f"Quota exceeded: {e}")
                time.sleep(60)  # Wait for some time before retrying
            else:
                logging.error(f"Error processing dataset {dataset_name}: {e}")

if __name__ == "__main__":
    main()
