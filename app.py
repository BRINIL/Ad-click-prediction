import uvicorn
from fastapi import FastAPI
from ads import ADs
import numpy as np
import pickle
import pandas as pd

app =FastAPI()
pickle_in= open('model_pkl.pkl','rb')
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message':'Hello, world'}

@app.get("/{name}")
def get_name(name:str):
    return {'message':f'Hello, {name}'}

@app.post('/predict')
def prediction(data:ADs):
    data = data.dict()
    age=data['age']
    EstimatedSalary=data['EstimatedSalary']
    gender=data['gender']
    print(classifier.predict([[age,EstimatedSalary,gender]]))
    pred = classifier.predict([[age,EstimatedSalary,gender]])
    if(pred[0]>0.5):
        pred='Ad clicked'
    else:
        pred='Ad not clicked'
    return {
        'pred':pred
    }

if __name__ == '__main__':
    uvicorn.run(app,host)
