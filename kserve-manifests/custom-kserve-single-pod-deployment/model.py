import os
import joblib
import urllib.request
import pandas as pd
from datetime import datetime
from kserve import ModelServer, Model

BASE = "https://github.com/vimalpillai21/house-price-pred-mlops/releases/download/kserve-pull"
FILES = {
    "preprocessor.pkl": f"{BASE}/preprocessor.pkl",
    "model.pkl": f"{BASE}/house_price_model.pkl",
}
ARTIFACT_DIR = "/tmp/artifacts"

class PreprocessorAndPredict(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.preprocessor = None
        self.model = None
        self.load()

    def load(self):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        for fname, url in FILES.items():
            dest = os.path.join(ARTIFACT_DIR, fname)
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest)  # follows GitHub's redirect
        self.preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        self.model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
        self.ready = True

    def predict(self, payload, headers=None):
        df = pd.DataFrame(payload["instances"])
        df['house_age'] = datetime.now().year - df['year_built']
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
        df['price_per_sqft'] = 0 
        X = self.preprocessor.transform(df)
        if hasattr(X,"toarray"):
            X = X.toarray()
        return X

    def predict(self, X, headers=None):
        return {"predictions": self.model.predict(X.tolist())}

if __name__ == "__main__":
    ModelServer().start([PreprocessorAndPredict("house-price-model")])
        
    



    