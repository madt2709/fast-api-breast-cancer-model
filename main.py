import pathlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load

app = FastAPI(title="Breast Cancer Prediction")
v1 = FastAPI()

# load model
model = load(pathlib.Path("model/breast-cancer-v1.joblib"))


class InputData(BaseModel):
    """
    Input data for /score endpoint
    """
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

    class Config:
        """
        Example input
        """
        schema_extra = {
            "example": {
                "mean_radius": 0.11663366574708534,
                "mean_texture": 0.8311471911073434,
                "mean_perimeter": 0.02972682563054485,
                "mean_area": 0.9692971491426847,
                "mean_smoothness": 0.9908881407484667,
                "mean_compactness": 0.4598655002542681,
                "mean_concavity": 0.435071538596121,
                "mean_concave": 0.30376330326515655,
                "mean_symmetry": 0.40295373742800045,
                "mean_fractal": 0.8955751899119784,
                "radius_error": 0.15532310443992947,
                "texture_error": 0.8748241716056037,
                "perimeter_error": 0.8975356513805876,
                "area_error": 0.6798507029550903,
                "smoothness_error": 0.5545973399577004,
                "compactness_error": 0.08596726038645586,
                "concavity_error": 0.1194602290932385,
                "concave_points": 0.33274109072680036,
                "symmetry_error": 0.6328850187510426,
                "fractal_dimension": 0.8365856677875979,
                "worst_radius": 0.07843848759751382,
                "worst_texture": 0.5993971591325171,
                "worst_perimeter": 0.6970005153118283,
                "worst_area": 0.6987544638881565,
                "worst_smoothness": 0.21504482049062845,
                "worst_compactness": 0.44208638261607736,
                "worst_concavity": 0.05774908876311857,
                "worst_concave": 0.20992526734817663,
                "worst_symmetry": 0.501663537181295,
                "worst_fractal": 0.13511891734039916
            }
        }


class ScoreOutputData(BaseModel):
    """
    Output data for /score endpoint
    """
    score: float


@v1.post("/score/", response_model=ScoreOutputData, status_code=201)
def score(data: InputData):
    """
    API endpoint which returns scores for breast cancer model. 
    As all inputs are measurements, we expect positive values for all inputs.
    """
    # raise an error if any values are negative
    negative_values = {k: v for k, v in data.dict().items() if v < 0}
    if negative_values:
        raise HTTPException(
            status_code=400, detail=f"All values should be positive: {negative_values}")
    # make prediction
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, -1]
    return {'score': result}


app.mount("/api/v1", v1)
