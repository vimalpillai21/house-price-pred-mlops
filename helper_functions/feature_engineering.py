import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("feature-engineering")

def create_features(df):
    """Creates new features from exisitng data"""

    logger.info("Creating new features")

    # Make copy to avoid modifying the original dataframe
    df_featured = df.copy()

    # Calculate house age
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df['year_built']
    logger.info("Creates 'house_age' feature")

    # Price per square foot
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqrt']
    logger.info("Created 'price_per_sqft' feature")

    # Bedroom to Bathroom ratio
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']

    # Handle division by zero
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf,-np.inf],np.nan)
    df_featured['bed_bath_ratio'].fillna(0)
    logger.info("Created 'bed_bath_ratio' feature")

    return df_featured