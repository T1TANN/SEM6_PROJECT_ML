import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

# uvicorn model:app --host 0.0.0.0 --port 8000 {commond to be executed}

class predict(BaseModel):
    pregnancies: int
    glucose: float
    bloodpressure: float
    skinthickness: float
    insulin: float
    bmi: float
    diabetespedigreefunction: float
    age: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

with open("../svc.pkl", "rb") as f:
    clf = pickle.load(f)


#Index
@app.get('/')
def index():
    return {'message': 'Welcome to Diabetes Prediction API'}

@app.post("/predict")
async def predict_diabetes(data: predict):
    try:
        sc = StandardScaler()
        data_list = pd.DataFrame([data.pregnancies, data.glucose, data.bloodpressure,
                      data.skinthickness, data.insulin, data.bmi,
                      data.diabetespedigreefunction, data.age])
        
        # Standardise the input data
        standardised_input = sc.fit_transform(data_list)

        # Make prediction using a trained model (clf)
        results = int(clf.predict(standardised_input.reshape(1, -1)))
        
        if results == 0:
            return "You are Healthy."
        elif results == 1:
            return "You are in Pre-Diabetic Stage."
    
        # return json.dumps(results)
    
    except Exception as e:
        # Return an appropriate error message if an exception occurs
        return {"error": str(e)}