import pickle
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
model_filename = 'app/stroke_model_LogReg_scikit.sav'


class PatientDataModel(BaseModel):
    age: int
    hypertension: int
    heart_disease: int
    ever_married: int
    avg_glucose_level: float


@app.get('/')
async def root():
    return {'Message': 'Stroke prediction app.'}


async def preprocess_data(data: Dict) -> List:
    age = data['age']
    hypertension = data['hypertension']
    heart_disease = data['heart_disease']
    ever_married = data['ever_married']
    avg_glucose_level = data['avg_glucose_level']
    return [age, hypertension, heart_disease, ever_married, avg_glucose_level]


@app.post('/predict')
async def predict(patient_data: PatientDataModel) -> Dict:
    data = patient_data.dict()
    data_processed = await preprocess_data(data=data)
    loaded_model = pickle.load(open(model_filename, 'rb'))
    try:
        proba = loaded_model.predict_proba([data_processed])
        probability = proba[0][1]
        prediction = 'Stroke probability is: {:.1%}'.format(probability)
        return {
            'prediction': prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
