import sys
import os
import argparse
import yaml
from config_loader import load_config
from llm_gpt4 import GPT4Model
from llm_gpt35 import GPT35Model
from logging_config import setup_logging
import logging
from pipeline_manager import run_pipeline

def get_available_options(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    available_models = ['gpt4', 'gpt3.5']
    available_datasets = list(config['datasets'].keys())
    return available_models, available_datasets

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Auto-Kaggle pipeline")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    parser.add_argument('--model', type=str, required=True, help="LLM model to use")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to use")
    
    # Parse known arguments to extract config path
    args, _ = parser.parse_known_args()
    
    # Get available options from config
    available_models, available_datasets = get_available_options(args.config)
    
    # Update argument parser with choices
    parser.add_argument('--model', type=str, required=True, choices=available_models, help="LLM model to use")
    parser.add_argument('--dataset', type=str, required=True, choices=available_datasets, help="Dataset to use")
    
    # Parse all arguments
    args = parser.parse_args()

    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    if args.model == 'gpt_4':
        llm = GPT4Model(api_key)
    elif args.model == 'gpt_3.5':
        llm = GPT35Model(api_key)
    else:
        raise ValueError("Invalid LLM model specified")

    # Extract dataset configuration
    if args.dataset not in config['datasets']:
        raise ValueError(f"Dataset {args.dataset} not found in configuration.")
    dataset_config = config['datasets'][args.dataset]

    # Run the pipeline
    run_pipeline(dataset_config, llm)

if __name__ == "__main__":
    main()
