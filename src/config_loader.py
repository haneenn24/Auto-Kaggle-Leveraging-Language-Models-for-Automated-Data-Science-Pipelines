import yaml
import logging

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Parameters:
    config_path (str): The path to the YAML configuration file.

    Returns:
    dict: The configuration dictionary.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = load_config("../config/config.yaml")
    print(config)
