from kfp import dsl, compiler
from kfp.dsl import Input, Output, Model, Dataset, Metrics

BASE_IMAGE = "python:3.11-slim"

# Data Preprocessing step
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4"]
)
def preprocess_data(
    raw_data_path: str,
    cleaned_data: Output[Dataset]
):
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("data-preprocessor")

    logger.info(f"Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded data with shape: {df.shape}")

    df_cleaned = df.copy()

    # Handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                fill_value = df_cleaned[column].median()
                kind = "median"
            else:
                fill_value = df_cleaned[column].mode()[0]
                kind = "mode"
            df_cleaned[column] = df_cleaned[column].fillna(fill_value)
            logger.info(f"Filled missing values in {column} with {kind}: {fill_value}")

    # Remove price outliers (IQR)
    q1 = df_cleaned["price"].quantile(0.25)
    q3 = df_cleaned["price"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    mask = (df_cleaned["price"] >= lower) & (df_cleaned["price"] <= upper)
    n_outliers = int((~mask).sum())
    if n_outliers:
        logger.info(f"Found {n_outliers} outliers in price column")
        df_cleaned = df_cleaned[mask]
        logger.info(f"Removed outliers. New dataset shape: {df_cleaned.shape}")

    df_cleaned.to_csv(cleaned_data.path, index=False)
    cleaned_data.metadata["rows"] = int(df_cleaned.shape[0])
    cleaned_data.metadata["columns"] = int(df_cleaned.shape[1])
    logger.info(f"Saved processed data to {cleaned_data.path}")


# Feature engineering step
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas==2.2.2", "numpy==1.26.4",
                         "scikit-learn==1.5.1", "joblib==1.4.2"]
)
def feature_engineering(
    cleaned_data: Input[Dataset],
    featured_data: Output[Dataset],
    preprocessor_artifact: Output[Model]
):
    import logging
    from datetime import datetime

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("feature-engineering")

    df = pd.read_csv(cleaned_data.path)

    # --- create_features ---
    df_featured = df.copy()
    df_featured["house_age"] = datetime.now().year - df["year_built"]
    df_featured["price_per_sqft"] = df_featured["price"] / df_featured["sqft"]
    df_featured["bed_bath_ratio"] = (
        (df_featured["bedrooms"] / df_featured["bathrooms"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    logger.info(f"Created featured dataset with shape: {df_featured.shape}")

    # --- create_preprocessor ---
    categorical_features = ["location", "condition"]
    numerical_features = ["sqft", "bedrooms", "bathrooms",
                            "house_age", "price_per_sqft", "bed_bath_ratio"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="mean"))]),
                numerical_features),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features),
        ]
    )

    X = df_featured.drop(columns=["price"], errors="ignore")
    y = df_featured["price"] if "price" in df_featured.columns else None
    X_transformed = preprocessor.fit_transform(X)
    if hasattr(X_transformed, "toarray"):  # in case the output is sparse
        X_transformed = X_transformed.toarray()
    logger.info("Fitted the preprocessor and transformed the features")

    joblib.dump(preprocessor, preprocessor_artifact.path)
    preprocessor_artifact.metadata["framework"] = "sklearn"

    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed["price"] = y.values
    # df_transformed.to_csv(featured_data.path, index=False)
    df_transformed.to_csv(featured_data.path, index=False)
    logger.info(f"Saved fully preprocessed data to {featured_data.path}")


# Train + Register model in MLFlow
@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=[ "pandas==2.2.2", "numpy==1.26.4", "scikit-learn==1.5.1",
            "xgboost==2.1.1", "mlflow==2.16.0", "joblib==1.4.2", "skops",]
)
def train_and_register(
    featured_data: Input[Dataset],
    model_config: dict,
    mlflow_tracking_uri: str,
    trained_model: Output[Model],
    metrics: Output[Metrics]
):
    import logging
    import platform

    import joblib
    import mlflow
    import mlflow.exceptions
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost as xgb
    from mlflow import MlflowClient
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("train-model")

    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor,
    }

    model_name = model_config["name"]
    algo = model_config["best_model"]
    params = model_config.get("parameters", {}) or {}
    target = model_config["target_variable"]

    if algo not in model_map:
        raise ValueError(f"Unsupported model: {algo}")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(model_name)

    data = pd.read_csv(featured_data.path)
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_map[algo](**params)

    with mlflow.start_run(run_name="final_training") as run:
        logger.info(f"Training model: {algo}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae, "r2": r2})
        metrics.log_metric("mae",mae)
        metrics.log_metric("r2",r2)
        

        mlflow.sklearn.log_model(
            model, "tuned_model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
                "collections.OrderedDict",
            ],
        )
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/tuned_model"

        # Save model as a KFP artifact
        joblib.dump(model, trained_model.path)
        trained_model.metadata.update({"algorithm":algo, "mae":mae,"r2": r2})

        # Register
        logger.info("Registering model to MLflow model registry")
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.RestException:
            pass  # already exists

        version = client.create_model_version(
            name=model_name, source=model_uri, run_id=run_id
        )
        client.set_registered_model_alias(
            name=model_name, version=version.version, alias="Champion"
        )

        description = (
            f"Model for predicting house prices.\n"
            f"Algorithm: {algo}\n"
            f"Hyperparameters: {params}\n"
            f"Features used: All features in the dataset except the target variable\n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {featured_data.uri}\n"
            f"Model saved at: {trained_model.uri}\n"
            f"Performance metrics:\n"
            f"  - MAE: {mae:.2f}\n"
            f"  - R²: {r2:.4f}"
        )
        client.update_registered_model(name=model_name, description=description)

        tags = {
            "algorithm": algo,
            "hyperparameters": str(params),
            "features": "All features except target variable",
            "target_variable": target,
            "training_dataset": featured_data.uri,
            "model_path": trained_model.uri,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        }
        for k, v in tags.items():
            client.set_registered_model_tag(model_name, k, v)

        logger.info(f"Final MAE: {mae:.2f}, R2: {r2:.4f}") 


# Pipeline
@dsl.pipeline(
    name="house-price-pipeline",
    description="Preprocess, feature engineering, train and register model"
)
def house_price_pipeline(
    raw_data_path: str,
    model_config: dict,
    mlflow_tracking_uri: str
):
    preprocess_task = preprocess_data(raw_data_path=raw_data_path)

    fe_task = feature_engineering(
        cleaned_data=preprocess_task.outputs["cleaned_data"],
        )

    train_task = train_and_register(
        featured_data=fe_task.outputs["featured_data"],
        model_config=model_config,
        mlflow_tracking_uri=mlflow_tracking_uri
    )

    train_task.set_caching_options(False)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/model_config.yaml")
    parser.add_argument("--output", default="house_price_pipeline.yaml")
    parser.add_argument("--submit", action="store_true",
                        help="Also submit a run to a KFP endpoint")
    parser.add_argument("--host", default=None, help="http://localhost:7000")
    parser.add_argument("--raw-data-path", default="https://github.com/vimalpillai21/house-price-pred-mlops/releases/download/kserve-skops/house_data.csv")
    parser.add_argument("--mlflow-tracking-uri",
                        default="http://mlflow.mlflow.svc.cluster.local:80")
    args = parser.parse_args()

    compiler.Compiler().compile(house_price_pipeline,args.output)
    print(f"Compiled pipeline to {args.output}")

    if args.submit:
        import kfp

        with open(args.config) as f:
            model_cfg = yaml.safe_load(f)["model"]

        client = kfp.Client(host=args.host)
        run = client.create_run_from_pipeline_package(
            args.output,
            arguments={
                "raw_data_path": args.raw_data_path,
                "model_config": model_cfg,
                "mlflow_tracking_uri": args.mlflow_tracking_uri
            }
        )
        print(f"Submitted run: {run.run_id}")
