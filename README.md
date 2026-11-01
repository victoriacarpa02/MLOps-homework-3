# MLOps Homework 3

I used Prefect to build a training pipeline and MLflow to save and register the model.

The pipeline loads the data, prepares it, trains a linear regression model and saves the results.

## Data

NYC Yellow Taxi trips for March 2023.

The model uses pickup and dropoff locations as separate features.

## Results

Q1: Prefect

Q2: Version 3.9.1

Q3: 3,403,766 loaded records

Q4: 3,316,216 records after filtering

Q5: Intercept 24.776397. Selected answer: 24.77

Q6: Model size 4,516 bytes. Selected answer: 4,534

Registered model: `nyc-yellow-taxi-duration`

Model version: `1`

## Setup

I used Python 3.13.12 and PowerShell on Windows.

Run all commands from the project folder. Install uv first if needed.

```powershell
uv venv --python 3.13.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
```

## Download data

```powershell
New-Item -ItemType Directory -Path .\data -Force

$url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
Invoke-WebRequest -Uri $url -OutFile ".\data\yellow_tripdata_2023-03.parquet"
```

## Start Prefect

```powershell
.\.venv\Scripts\prefect.exe config set PREFECT_API_URL=http://127.0.0.1:4200/api

.\.venv\Scripts\prefect.exe server start --host 127.0.0.1 --port 4200
```

Open http://127.0.0.1:4200 in a browser.

## Start MLflow

Open another terminal in the project folder.

```powershell
.\.venv\Scripts\mlflow.exe server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts --host 127.0.0.1 --port 5001
```

Open http://127.0.0.1:5001 in a browser.

## Run the pipeline

Keep both servers running. Open another terminal in the project folder.

```powershell
.\.venv\Scripts\python.exe .\pipeline.py
```

The results appear in the terminal and Prefect logs.

MLflow stores the model in the `yellow-taxi-training` experiment. The fitted DictVectorizer is saved as a separate artifact.

Each successful registration creates a new model version.

## Notes

Data, generated artifacts, the local MLflow database and the virtual environment are excluded from Git.

[Homework instructions](https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2025/03-orchestration/homework.md)