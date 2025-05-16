import sys
import os
import importlib.util
import base64
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline

# 📁 Base path to the Notebooks folder
BASE_DIR = os.path.join(os.getcwd(), "Notebooks")

# 🧠 Classifier name, import module, and model folder
classifiers = {
    "car": ("infer_car", os.path.join(BASE_DIR, "car_classifier", "final_model")),
    "fashion": ("infer_fashion", os.path.join(BASE_DIR, "fashion_classifier", "final_model")),
    "food": ("infer_food", os.path.join(BASE_DIR, "Food_classifier", "final_model")),
}

# 🧠 Load and patch inference functions
inference_funcs = {}

for name, (module_name, model_folder) in classifiers.items():
    # Import the inference script dynamically
    module_path_py = os.path.join(BASE_DIR, f"{name}_classifier", f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # ✅ Fix model path for Hugging Face compatibility
    model_path = Path(model_folder).resolve().as_posix()

    # ✅ Patch classifier into module using correct path
    module.classifier = pipeline("image-classification", model=model_path)

    # ✅ Patch a default inference function if missing
    def make_inference_func(clsfr):
        def inference(image):
            result = clsfr(image)
            best = max(result, key=lambda x: x['score'])
            return {"label": best['label'], "score": float(best['score'])}
        return inference

    inference_funcs[name] = make_inference_func(module.classifier)

# 🚀 Initialize FastAPI app
app = FastAPI(title="Product Classification API", version="1.0.0")

# 🧾 Define request body format
class RequestData(BaseModel):
    image: str  # base64-encoded string

# 📦 Decode and load image
async def process_image(req: RequestData) -> Image.Image:
    try:
        image_data = io.BytesIO(base64.b64decode(req.image))
        img = Image.open(image_data)
        img.load()
        print(f"✅ Image loaded: {img}")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"❌ Failed to process image: {str(e)}")

# 🤖 Call the corresponding model inference
async def handle_prediction(img: Image.Image, model_name: str) -> dict:
    try:
        inference_func = inference_funcs[model_name]
        prediction = inference_func(img)
        if prediction is None:
            raise HTTPException(status_code=500, detail="Prediction failed.")
        return {"model": model_name, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 📮 POST endpoint for prediction
@app.post("/predict_car/")
async def predict_car(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, "car")

@app.post("/predict_fashion/")
async def predict_fashion(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, "fashion")

@app.post("/predict_food/")
async def predict_food(req: RequestData):
    img = await process_image(req)
    return await handle_prediction(img, "food")

