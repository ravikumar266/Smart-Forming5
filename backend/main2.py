from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    'Corn___Northern_Leaf_Blight', 'Wheat___Yellow_Rust', 'Sugarcane_Bacterial Blight',
    'Potato___Healthy', 'Rice___Neck_Blast', 'Corn___Healthy', 'Wheat___Brown_Rust',
    'Corn___Gray_Leaf_Spot', 'Rice___Brown_Spot', 'Sugarcane_Red Rot', 'Sugarcane_Healthy',
    'Wheat___Healthy', 'Rice___Leaf_Blast', 'Potato___Late_Blight', 'Rice___Healthy',
    'Corn___Common_Rust', 'Potato___Early_Blight'
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((250, 250))  
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        all_predictions = {
            class_names[i]: float(predictions[i]) 
            for i in range(len(class_names))
        }

        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
