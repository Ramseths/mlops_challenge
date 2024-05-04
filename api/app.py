from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd

model = load("../models/trained_model-1.0.0.joblib")

class InputData(BaseModel):
    # Features
    floor_area: float
    interest: float
    cpi: float

app = FastAPI()

@app.get('/')
def home():
    return {"running": "OK"}

@app.post("/predict")
def predict(data: InputData):
    try:
        # Model inference
        input_data = pd.DataFrame({'Floor Area' : [data.floor_area],
                                'Interest' : [data.interest],
                                'CPI': [data.cpi]})
        print(input_data)
        prediction = model.predict(input_data)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
