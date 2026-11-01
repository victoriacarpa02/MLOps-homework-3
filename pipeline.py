import pandas as pd
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


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


@flow(name="nyc-taxi-training", log_prints=True)
def training_pipeline(
    data_path: str = "data/yellow_tripdata_2023-03.parquet",
):
    df = load_data(data_path)
    df = prepare_data(df)
    dv, model = train_model(df)

    return dv, model


if __name__ == "__main__":
    training_pipeline()