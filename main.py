# main.py
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.get("/predict/{value}")
def predict(value: float):
    data = np.array([[value]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}


    