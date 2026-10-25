import argparse
import json
import logging
import os
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and optionally register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory to save trained models")
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default=None,
        help="MLflow tracking URI. If omitted, MLflow is not used at all."
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow even if a tracking URI is provided."
    )
    return parser.parse_args()


def get_model_instance(name, params):
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'XGBoost': xgb.XGBRegressor
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    logger.info(f"Model: {name}, params: {params}")
    return model_map[name](**params)


def log_to_mlflow(args, model_cfg, model, target, mae, r2):
    """All MLflow logic lives here. mlflow is imported lazily so it does not
    need to be installed when MLflow is disabled."""
    import mlflow
    import mlflow.sklearn
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(model_cfg['name'])

    model_name = model_cfg['name']

    with mlflow.start_run(run_name="final_training"):
        run_id = mlflow.active_run().info.run_id

        # Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({'mae': mae, 'r2': r2})

        # Log model
        mlflow.sklearn.log_model(
            model, "tuned_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
                "collections.OrderedDict",
            ]
        )
        model_uri = f'runs:/{run_id}/tuned_model'

        logger.info("Registering model to MLflow model registry")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            # model already exists
            pass

        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )

        client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias="Champion"
        )

        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.model_dir}/trained/{model_name}.pkl\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        tags = {
            "algorithm": model_cfg['best_model'],
            "hyperparameters": str(model_cfg['parameters']),
            "features": "All features except target variable",
            "target_variable": target,
            "training_dataset": args.data,
            "model_path": f"{args.model_dir}/trained/{model_name}.pkl",
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in tags.items():
            client.set_registered_model_tag(model_name, k, v)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']
    model_name = model_cfg['name']

    use_mlflow = bool(args.mlflow_tracking_uri) and not args.no_mlflow
    logger.info(f"MLflow logging: {'enabled' if use_mlflow else 'disabled'}")

    # Load data
    data = pd.read_csv(args.data)
    target = model_cfg['target_variable']

    # Use all features except the target variable
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = get_model_instance(model_cfg['best_model'], model_cfg['parameters'])
    logger.info(f"Training model: {model_cfg['best_model']}")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    logger.info(f"Final MAE: {mae:.2f}, R2: {r2:.4f}")

    # Save model locally (always)
    # trained_dir = os.path.join(args.model_dir, "trained")
    os.makedirs(trained_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, f"{model_name}.pkl")
    joblib.dump(model, save_path)
    logger.info(f"Saved trained model to: {save_path}")

    # Save metrics locally (always) so CI can read/upload them
    metrics_path = os.path.join(trained_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mae": mae, "r2": r2}, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Optional MLflow logging
    if use_mlflow:
        log_to_mlflow(args, model_cfg, model, target, mae, r2)


if __name__ == "__main__":
    args = parse_args()
    main(args)