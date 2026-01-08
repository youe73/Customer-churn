import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import joblib

scaler = joblib.load("./model/scaler.joblib")

model = load_model("model/customer_churn_model.keras")

class InputData(BaseModel):
    Age:  int
    Gen_Female: int 
    Gen_Male: int
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: int 
    Contract_Length: int 
    Total_Spend: float
    Last_Interaction: int
 
feature_names = joblib.load("./model/feature_names.joblib")

app = FastAPI()

@app.get("/")
def main():
    return "Customer churn analysis"


@app.post("/prediction")
def prediction(data:InputData):        
    #input_data = pd.DataFrame.from_dict(data,orient='index').transpose()
    input_data = pd.DataFrame([data.model_dump()])
    input_data = input_data[feature_names]
    scaled_input_data = scaler.transform(input_data)
    y_predicted = model.predict(scaled_input_data)  
    if int(y_predicted[0][0]) == 0:
        return f"Label:{int(y_predicted[0][0])} - Customer will remain in the company's services"
    else:
        return f"Label:{int(y_predicted[0][0])} - Customer will discontinue the company's subscription"
    

