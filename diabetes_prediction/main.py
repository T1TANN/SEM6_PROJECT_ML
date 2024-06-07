import pickle
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from pydantic import BaseModel

def standard_scaler_to_array(data):
    # Convert the DataFrame to a NumPy array
    data_array = data.to_numpy()

    # Calculate the mean and standard deviation along each feature axis (axis 0)
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)

    # Avoid division by zero by adding a small epsilon if std is close to zero
    epsilon = 1e-10
    std = np.where(std < epsilon, epsilon, std)

    # Standardize the data
    standardized_array = (data_array - mean) / std

    return standardized_array

class predict(BaseModel):
    pregnancies: int = Query(..., description="Number of pregnancies", ge=0)
    glucose: float = Query(..., description="Glucose level", ge=0)
    bloodpressure: float = Query(..., description="Blood pressure", ge=0)
    skinthickness: float = Query(..., description="Skin thickness", ge=0)
    insulin: float = Query(..., description="Insulin level", ge=0)
    bmi: float = Query(..., description="Body Mass Index (BMI)", ge=0)
    diabetespedigreefunction: float = Query(..., description="Diabetes pedigree function")
    age: int = Query(..., description="Age", ge=0)

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


#Predict
@app.post("/predict")
async def predict_diabetes(data: predict):
    try:
        data_list = pd.DataFrame([data.pregnancies, data.glucose, data.bloodpressure,
                      data.skinthickness, data.insulin, data.bmi,
                      data.diabetespedigreefunction, data.age])
        
        # Standardise the input data
        standardised_input = standard_scaler_to_array(data_list)

        # Make prediction using a trained model (clf)
        results = int(clf.predict(standardised_input.reshape(1, -1)))
        
        if results == 0:
            return f"You are Healthy"
        else:
            return f"You have Diabetes"
    
        # return json.dumps(results)
    
    except Exception as e:
        # Return an appropriate error message if an exception occurs
        return {"error": str(e)}
 
    
    
    


