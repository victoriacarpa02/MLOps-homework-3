import pandas as pd
from prefect import flow, task


@task
def load_data(data_path: str):
    df = pd.read_parquet(data_path)

    print(f"Input file: {data_path}")
    print(f"Q3 - Loaded rows: {len(df):,}")

    return df


@flow(name="nyc-taxi-training", log_prints=True)
def training_pipeline(
    data_path: str = "data/yellow_tripdata_2023-03.parquet",
):
    df = load_data(data_path)
    return len(df)


if __name__ == "__main__":
    training_pipeline()