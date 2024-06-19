import logging

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        filename='pipeline.log',
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

# Example usage
if __name__ == "__main__":
    setup_logging()
    logging.info("Logging setup complete.")
