import os
# import joblib
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime
from kserve import ModelServer, Model
from skops.io import load as skops_load

BASE = "https://github.com/vimalpillai21/house-price-pred-mlops/releases/download/kserve-skops"
FILES = {
    "preprocessor.skops": f"{BASE}/preprocessor.skops",
    "model.skops": f"{BASE}/model.skops",
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
                try:
                    print(f"Downloading {url}....")
                    request = urllib.request.Request(
                        url=url,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )
                    with urllib.request.urlopen(request,timeout=60) as response:
                        with open(dest,"wb") as f:
                            f.write(response.read())
                    print(f"Downloaded successfully: {dest}")
                except urllib.error.HTTPError as e:
                    print(f"HTTP Error for {fname}: {e.code} - {e.reason}")
                except urllib.error.URLError as e:
                    print(f"URL Error for  {fname}: {e.reason}")
                except OSError as e:
                    print(f"OS Error for {fname}: {e.reason}")

        # self.preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
        # self.model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
        self.preprocessor = skops_load(
                os.path.join(ARTIFACT_DIR, "preprocessor.skops"),
                trusted=["numpy.dtype"],
            )
        self.model = skops_load(
            os.path.join(ARTIFACT_DIR, "model.skops"),
            trusted=["sklearn.tree._tree.Tree"],
        )
        self.ready = True

    def preprocess(self, payload, headers=None):
        df = pd.DataFrame(payload["instances"])
        df['house_age'] = datetime.now().year - df['year_built']
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
        df['price_per_sqft'] = 0 
        X = self.preprocessor.transform(df)
        if hasattr(X,"toarray"):
            X = X.toarray()
        return X

    def predict(self, X, headers=None):
        preds = self.model.predict(X)
        return {"predictions": np.asarray(preds).tolist()}

if __name__ == "__main__":
    ModelServer().start([PreprocessorAndPredict("house-price-model")])
        
    



    