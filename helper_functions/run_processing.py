import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('data-preprocessor')


def load_data(file_path):
    """Loads data from a CSV file."""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Cleans the dataset by handling missing values and outliers."""
    logger.info("Cleaning dataset")

    # Make copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value = df_cleaned[column].median()
                df_cleaned[column].fillna(median_value)
                logger.info(f"Filled missing values in {column} with median: {median_value}")
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled missing values in {column} with mode: {mode_value}")


    # Handle outliers using IQR method
    Q1 = df_cleaned['price'].quantile(0.25)
    Q3 = df_cleaned['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Find out extreme outliers
    outliers = df_cleaned[(df_cleaned['price'] < lower_bound) | 
                          (df_cleaned['price'] > upper_bound)]

    if not outliers.empty:
        logger.info(f"Found {len(outliers)} outliers in price column")
        df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & 
                                (df_cleaned['price'] <= upper_bound)]  
        logger.info(f"Removed outliers. New dataset shape: {df_cleaned.shape}")

    return df_cleaned

def process_data(input_file, output_file):
    """Full data processing pipeline."""
    # Make output directory if it does not exists
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(input_file)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Clean data
    df_cleaned = clean_data(df)

    # Save processed data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")
    
    return df_cleaned

if __name__ == "__main__":
    process_data(
        input_file = "data/raw/house_data.csv",
        output_file = "data/processed/cleaned_house_data.csv"
    )