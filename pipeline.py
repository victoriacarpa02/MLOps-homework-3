import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.models import Model


@task
def load_data(data_path: str):
    df = pd.read_parquet(data_path)

    print(f"Input file: {data_path}")
    print(f"Q3 - Loaded rows: {len(df):,}")

    return df


@task
def prepare_data(df: pd.DataFrame):
    df = df.copy()

    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[
        (df["duration"] >= 1) & (df["duration"] <= 60)
    ].copy()

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    print(f"Q4 - Prepared rows: {len(df):,}")

    return df


@task
def train_model(df: pd.DataFrame):
    categorical = ["PULocationID", "DOLocationID"]
    train_dicts = df[categorical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df["duration"].to_numpy()

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Q5 - Model intercept: {model.intercept_:.6f}")

    return dv, model


@task
def save_model(dv, model, data_path: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("yellow-taxi-training")

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("features", "PULocationID,DOLocationID")
        mlflow.log_metric("intercept", float(model.intercept_))

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        dv_path = output_dir / "dv.pkl"
        with dv_path.open("wb") as file:
            pickle.dump(dv, file)

        mlflow.log_artifact(
            str(dv_path),
            artifact_path="preprocessor",
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

        model_metadata = Model.load(model_info.model_uri)

        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name="nyc-yellow-taxi-duration",
        )

        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")
        print(f"Q6 - model_size_bytes: {model_metadata.model_size_bytes}")
        print(f"Registered model: {registered_model.name}")
        print(f"Model version: {registered_model.version}")

        return model_info.model_uri


@flow(name="nyc-taxi-training", log_prints=True)
def training_pipeline(
    data_path: str = "data/yellow_tripdata_2023-03.parquet",
):
    df = load_data(data_path)
    df = prepare_data(df)
    dv, model = train_model(df)
    model_uri = save_model(dv, model, data_path)

    return model_uri


if __name__ == "__main__":
    training_pipeline()