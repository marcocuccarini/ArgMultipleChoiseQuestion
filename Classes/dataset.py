import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from typing import Optional

class MedQADatasetLoader:
    def __init__(self, dataset: str = "moaaztameer/medqa-usmle"):
        """
        Initialize the dataset loader.

        Args:
            dataset (str): The Kaggle dataset identifier.
        """
        self.dataset = dataset
        self.df: Optional[pd.DataFrame] = None

    def load(self, file_path: str = "", **pandas_kwargs) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.

        Args:
            file_path (str): Path inside the dataset to the file you want to load.
            pandas_kwargs: Extra arguments to pass to pandas read function.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        self.df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            self.dataset,
            file_path,
            pandas_kwargs=pandas_kwargs
        )
        return self.df

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return the first n rows of the dataset.

        Args:
            n (int): Number of rows to display.

        Returns:
            pd.DataFrame
        """
        if self.df is None:
            raise ValueError("Dataset not loaded yet. Call load() first.")
        return self.df.head(n)


