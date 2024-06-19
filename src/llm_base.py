from abc import ABC, abstractmethod

class LLMBase(ABC):
    @abstractmethod
    def generate_code(self, dataset_path):
        """
        Generate code for the entire machine learning pipeline based on the dataset provided.

        Parameters:
        dataset_path (str): Path to the dataset file.

        Returns:
        str: Generated code as a string.
        """
        pass
