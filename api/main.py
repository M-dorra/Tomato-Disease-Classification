from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app=FastAPI()

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL=tf.keras.models.load_model("../tomatoes.keras")

CLASS_NAMES=["Bacterial Spot","Early Blight","Late Blight","Leaf Mold","Septoria Leaf Spot","Spider Mites Two Spotted Spider Mite","Target Spot","YellowLeaf Curl Virus","Mosaic Virus","Healthy"]

# @app.get("/")
# async def ping():
#     return "Hello I am alive"

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile):
    bytes=await file.read()
    image=read_file_as_image(bytes)
    image_batch=np.expand_dims(image,0)
    prediction=MODEL.predict(image_batch)
    index=np.argmax(prediction[0])
    predicted_class=CLASS_NAMES[index]
    confidence=np.max(prediction[0])
    return{
        'class':predicted_class,
        'confidence':float(confidence)
    }
    
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000) 