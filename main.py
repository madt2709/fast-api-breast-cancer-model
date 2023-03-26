from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib 

app = FastAPI(title = 'Breast Cancer Prediction')

# load model
model = load(pathlib.Path('model/breast-cancer-v1.joblib'))

class InputData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave: float
    mean_symmetry: float
    mean_fractal: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points: float
    symmetry_error: float
    fractal_dimension: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave: float
    worst_symmetry: float
    worst_fractal: float

class ScoreOutputData(BaseModel):
    score:float

@app.post('/score', response_model = ScoreOutputData)
def score(data:InputData):
    model_input = np.array([v for k,v in data.dict().items()]).reshape(1,-1)
    result = model.predict_proba(model_input)[:,-1]
    
    return {'score':result}

