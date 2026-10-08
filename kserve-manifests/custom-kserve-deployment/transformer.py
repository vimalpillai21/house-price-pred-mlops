import argparse
import joblib
import pandas as pd
from datetime import datetime
from kserve import Model, ModelServer, model_server

class Preprocessor(Model):
    def __init__(self, name, predictor_host, protocol):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.preprocessor = joblib.load("/app/preprocessor.pkl")
        self.ready = True

    def preprocess(self, payload, headers=None):
        df = pd.DataFrame(payload["instances"])
        df['house_age'] = datetime.now().year - df['year_built']
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
        df['price_per_sqft'] = 0 
        X = self.preprocessor.transform(df)
        if hasattr(X,"toarray"):
            X = X.toarray()
        return {"instances": X.tolist()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    args, _ = parser.parse_known_args()
    model = Preprocessor(args.model_name, args.predictor_host, args.protocol)
    ModelServer().start([model])