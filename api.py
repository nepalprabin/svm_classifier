import io
import base64
import json
import time
from tempfile import NamedTemporaryFile
from pathlib import Path
import shutil
from numpy import asarray

from fastapi import UploadFile
from typing import List
import fastapi
from fastapi import File
from fastapi import requests
from starlette.requests import Request
import uvicorn
from skimage.transform import resize
from skimage.io import imread

import pickle

from PIL import Image

import numpy as np

app = fastapi.FastAPI()

# Model
def get_model():
    model = pickle.load(open('img_model.p', 'rb'))
    return model

CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Model Prediction
def predict_res(model, img_path):
    img = imread(img_path)
    img_resize=resize(img,(28,28,3))
    flatten_image = [img_resize.flatten()]
    probability=model.predict_proba(flatten_image)
    results = []
    for ind,val in enumerate(CATEGORIES):
        results.append({'label':val,'probability':probability[0][ind]})
        #  print(f'{val} = {probability[0][ind]*100}%')
    return results
    # print("The predicted image is : "+CATEGORIES[model.predict(flatten_image)[0]])

# get model
model = get_model()


@app.get('/')
def index():
    return {"Home"}


@app.post('/api/predict')
def predict(request: Request, sample_image: UploadFile = File(...)):
    data = {'success': False}
    
    if request.method == 'POST':
        image_path = str(save_upload_file_tmp(sample_image))
        res = predict_res(model, image_path)
        return {'result': res }
    # return sample_image

    
def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
        tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='127.0.0.1')