from pathlib import Path
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import mlflow.sklearn
import pandas as pd

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"

MODEL_PATH = BASE_DIR / "models" / "best_registry_model_18-08-2026"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="LungAI API",
    description="Machine Learning API for Lung Cancer Classification",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD ML MODEL
# ============================================================

try:

    model = mlflow.sklearn.load_model(
        str(MODEL_PATH)
    )

    print("ML model loaded successfully.")

except Exception as e:

    model = None

    print("Failed to load MLflow model:")
    print(e)


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", include_in_schema=False)
def serve_frontend():

    return FileResponse(FRONTEND_PATH)


# ============================================================
# INPUT SCHEMA
# ============================================================

class LungCancerInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    GENDER: str
    AGE: int
    SMOKING: str
    YELLOW_FINGERS: str
    ANXIETY: str
    PEER_PRESSURE: str
    CHRONIC_DISEASE: str = Field(alias="CHRONIC DISEASE")
    FATIGUE: str
    ALLERGY: str
    WHEEZING: str
    ALCOHOL_CONSUMING: str = Field(alias="ALCOHOL CONSUMING")
    COUGHING: str
    SHORTNESS_OF_BREATH: str = Field(alias="SHORTNESS OF BREATH")
    SWALLOWING_DIFFICULTY: str = Field(alias="SWALLOWING DIFFICULTY")
    CHEST_PAIN: str = Field(alias="CHEST PAIN")

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "message": "LungAI API is running",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: LungCancerInput):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="ML model could not be loaded."
        )

    try:

        # ------------------------------------------------------
        # Create DataFrame with EXACT training column names
        # ------------------------------------------------------

        input_data = pd.DataFrame([
            {
                "GENDER": data.GENDER,
                "AGE": data.AGE,
                "SMOKING": data.SMOKING,
                "YELLOW_FINGERS": data.YELLOW_FINGERS,
                "ANXIETY": data.ANXIETY,
                "PEER_PRESSURE": data.PEER_PRESSURE,
                "CHRONIC DISEASE": data.CHRONIC_DISEASE,
                "FATIGUE ": data.FATIGUE,
                "ALLERGY ": data.ALLERGY,
                "WHEEZING": data.WHEEZING,
                "ALCOHOL CONSUMING": data.ALCOHOL_CONSUMING,
                "COUGHING": data.COUGHING,
                "SHORTNESS OF BREATH": data.SHORTNESS_OF_BREATH,
                "SWALLOWING DIFFICULTY": data.SWALLOWING_DIFFICULTY,
                "CHEST PAIN": data.CHEST_PAIN
            }
        ])

        print("\nReceived input:")
        print(input_data)
        print("\n========== INPUT DATA ==========")
        print(input_data)
        print("\n========== DATA TYPES ==========")
        print(input_data.dtypes)
        print("\n========== MODEL ==========")
        print(model)

        # ------------------------------------------------------
        # Prediction
        # ------------------------------------------------------

        prediction = model.predict(input_data)[0]

        # ------------------------------------------------------
        # Probability
        # ------------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability = float(
                probabilities[int(prediction)]
            )

        # ------------------------------------------------------
        # Result
        # ------------------------------------------------------

        if int(prediction) == 1:

            result_message = "Lung cancer risk detected."

        else:

            result_message = "No lung cancer risk detected."

        return {

            "prediction": int(prediction),

            "probability": probability,

            "message": result_message

        }

    except Exception as e:

        print("\n========== FULL ERROR ==========")
        traceback.print_exc()
        print("================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )