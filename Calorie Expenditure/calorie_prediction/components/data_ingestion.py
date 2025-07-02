import csv
import pandas as pd
import logging
from typing import Optional, Dict, Any
import os

# Configure basic logging for demonstration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CSVDataLoader:
    def __init__(self):
        pass

    def read_csv_to_dataframe(self, file_path, file_name) -> pd.DataFrame:
        if not os.path.exists(file_path + file_name):
            logging.error(f"File not found: {file_path + file_name}")
            raise FileNotFoundError(f"File not found: {file_path + file_name}")
        
        df = pd.read_csv(file_path + file_name)
        logging.info(f"Successfully read CSV file: {file_path + file_name}")
        return df