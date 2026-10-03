import joblib
import pandas as pd
from datetime import datetime
from schemas import HousePricePredictionRequest, PredictionResponse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Load model and preprocessor
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    logger.error(f"Exception occured - {str(e)}")
    raise e

def predict_price(request: HousePricePredictionRequest) -> PredictionResponse:
    """Predict house price based on input features"""

    # Prepare input data
    input_data = pd.DataFrame([request.model_dump()])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0 # add dummy value

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(processed_features)[0]

    # Convert numpy float32 to Python float and round to 2 decimal places
    predicted_price = round(float(predicted_price),2)

    # Calculate confidence interval
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )


def batch_predict(requests: list[HousePricePredictionRequest]) -> list[float]:
    """Perform batch predictions"""
    input_data = pd.DataFrame([req.model_dump() for req in requests])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make Predictions
    predictions = model.predict(processed_features)
    return predictions.tolist()