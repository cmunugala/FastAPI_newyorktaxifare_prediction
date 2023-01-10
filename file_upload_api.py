from fastapi import FastAPI,File,UploadFile
import pandas as pd
import numpy as np
from model.model import NeuralNetwork
from model.predict import predict,preprocess
from config import CONFIG
import torch
from joblib import load
from pydantic import BaseModel, Field
import json


app = FastAPI()

@app.on_event("startup")
async def startup_event():

    # Initialize the pytorch model
    model = NeuralNetwork(10,10,5,5,1)
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "scaler": load(CONFIG['SCALAR_PATH']),  # joblib.load
        "model": model
    }

@app.get("/")
def home():
    return {'health check':'OK','model_version':'v1'}

@app.post("/prediction")
#async def upload_file(file:UploadFile = File(...)):
async def upload_file_predict(file:UploadFile = UploadFile(...)):
    
    df = pd.read_csv(file.file,sep='\t',header=None).T.to_dict()

    preds = []
    #iterate through rows of input file and save predictions in list:preds
    for i in range(0,len(df)):
        X = np.array(list(df[i].values()))

        #model inference and append preds to prediction list to return
        y_pred = predict(app.package,[X])
        preds.append(float(y_pred[0][0]))
    
    return {'prediction':json.dumps(preds)}



