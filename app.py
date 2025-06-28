import sys
import os

import certifi
ca=certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")

import pymongo
from NetworkSecurityFun.exception.exception import NetworkSecurityException
from NetworkSecurityFun.logging.logger import logger
from NetworkSecurityFun.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File,UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from NetworkSecurityFun.utils.main_utils.utils import load_object

from NetworkSecurityFun.utils.ml_utils.model.estimator import NetworkSecurityModel

client=pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

from NetworkSecurityFun.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from NetworkSecurityFun.constants.training_pipeline import DATA_INGESTION_DATABASE_NAME

database=client[DATA_INGESTION_DATABASE_NAME]
collection=database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load objects
        preprocessor = load_object("final_models/preprocessor.pkl")
        final_model = load_object("final_models/model.pkl")
        network_model = NetworkSecurityModel(preprocessor=preprocessor, model=final_model)

        expected_columns = list(preprocessor.feature_names_in_)
        actual_columns = list(df.columns)

        print("Expected columns by model:", expected_columns)
        print("Columns in uploaded CSV:", actual_columns)

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

        df_for_pred = df[expected_columns].copy()

        y_pred = network_model.predict(df_for_pred)

        df['predicted_column'] = y_pred

        df.to_csv('prediction_output/output.csv', index=False)
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
if __name__ == "__main__":
    app_run("app:app", host="0.0.0.0", port=8000)