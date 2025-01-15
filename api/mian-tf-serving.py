#Download tensorflow,fastapi,pillow,python-serving-apy,,uvicorn,python-multipart,numpy,matplotlib     First!!!!!
##docker run -t --rm -p 8502:8502 -v D:/year3/self_study_DL:/self_study_DL tensorflow/serving --rest_api-PORT=8502 --model_config_file=self_study_DL/models.config
from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
app=FastAPI()
origin=[
    'http://localhost',
    'http://localhost:3000',
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)
endpoint='http://localhost:8502/v1/models/potatoes_model:predict'

MODEL=tf.keras.models.load_model("D:/year3/self_study_DL/all_models/1.keras")
CLASS_NAMES=['Early Blight','Late Blight','Healthy']
def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile=File(...)
):
    bytes=await file.read()
    image=read_file_as_image(bytes)
    img_batch=np.expand_dims(image,0)
    json_data={
        'instances':img_batch.tolist()
    }
    response=requests.post(endpoint,json=json_data)
    prediction=np.array(response.json()['predictions'][0])
    # prediction=MODEL.predict(img_batch)
    predict_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)
    return {
        'class': predict_class,
        'confidence': float(confidence)
    }
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8001)