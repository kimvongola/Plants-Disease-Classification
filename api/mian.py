#Download tensorflow,fastapi,pillow,python-serving-apy,,uvicorn,python-multipart,numpy,matplotlib     First!!!!!
from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL=tf.keras.models.load_model("../self_study_DL/all_models/1.keras")
MODEL.export("../self_study_DL/all_models/1")

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
    prediction=MODEL.predict(img_batch)
    predict_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    return {
        'class': predict_class,
        'confidence': float(confidence)
    }
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8001)