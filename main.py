#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model (make sure to replace with your actual model file path)
model = joblib.load('model.pkl')

# Create FastAPI app instance
app = FastAPI()

# Define the schema for the incoming request body
class RequestBody(BaseModel):
    bedrooms: int
    bathrooms: int
    verified: int
    sizeMin: int
    furnishing_NO: int
    furnishing_PARTLY: int
    furnishing_YES: int

@app.post("/predict/")
def predict(request: RequestBody):
    # Convert incoming data to a NumPy array for prediction
    features = np.array([[request.bedrooms, request.bathrooms, request.verified, 
                          request.sizeMin, request.furnishing_NO, request.furnishing_PARTLY, request.furnishing_YES]])
    
    # Predict the log price
    log_price = model.predict(features)
    
    # Convert log price back to original scale
    price = np.exp(log_price)[0]
    
    return {"predicted_price": price}

