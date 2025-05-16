# 🧾 Multi-Product Classifier

A FastAPI-based image classification API that categorizes product images into distinct domains such as cars, fashion, and food. Powered by Hugging Face Transformers and dynamically loaded checkpoints for each domain model.

---

## 🧠 Overview

This project supports multi-domain product classification by dynamically importing and serving separate inference modules. Each module uses a fine-tuned Hugging Face model for its respective domain and is deployed using a clean FastAPI interface.

---

## 📁 Project Structure

```
Multi-Product_classifier/
├── main.py # FastAPI app for all endpoints
├── requirements.txt # Python dependencies
├── README.md
└── Notebooks/
├── car_classifier/
│ ├── infer_car.py # Inference logic for car classification
│ └── car_models_image_classifier/
│ └── checkpoint-37500/
├── fashion_classifier/
│ ├── infer_fashion.py
│ └── fashion_model/
│ └── checkpoint-2000/
└── food_classifier/
├── infer_food.py
└── results/
└── checkpoint-1500/
```

---

## 📌 Features

- 🧠 Domain-specific image classification using Hugging Face `pipeline()`
- ⚙️ Dynamic module loading for flexibility and scalability
- 🔗 FastAPI server with custom endpoints for each classifier
- 💡 Support for JPEG/PNG image input via base64-encoded strings
- ✅ Returns most confident label and score per image

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MarwanAbdellah/Multi-Product_classifier.git
cd Multi-Product_classifier
```
### 2. Create a Virtual Environment
```
python -m venv venv
```
# On Windows:
```
venv\Scripts\activate
```
# On macOS/Linux:
```
source venv/bin/activate
```
### 3. Install Requirements
```
pip install -r requirements.txt
```
▶️ Run the App
```
uvicorn main:app --reload
```
Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI.

## 🧪 API Endpoints

Each endpoint takes a base64-encoded image and returns the top predicted class with its confidence score.

### POST `/predict_car/`
### POST `/predict_fashion/`
### POST `/predict_food/`

---

### ✅ Example Request Body

```json
{
  "image": "base64_encoded_image_string"
}
```
---

### ✅ Example Response
```json
{
  "model": "car",
  "prediction": {
    "label": "Sedan",
    "score": 0.9876
  }
}
```
---

### ✅ Example with curl
```
curl -X POST http://127.0.0.1:8000/predict_car/ \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"<base64-encoded-image>\"}"
```
---

### 📦 Requirements
```
fastapi
uvicorn
transformers
pydantic
pillow
numpy
python-dotenv
```
---

### Install with:
```
pip install -r requirements.txt
```
---

### 🔐 Environment & Deployment Notes
* .env is not required unless you add external API services.

* Make sure to include the following in .gitignore:
```
__pycache__/
.venv/
.env
*.ckpt

```
---

### 📸 Image Input Format
* Accepts `.jpg`, `.jpeg`, `.png` files
* Must be base64-encoded before sending in POST body
To convert an image to base64 in Python:
```
import base64
with open("image.jpg", "rb") as img:
    b64_str = base64.b64encode(img.read()).decode()
```
---

### 📜 License
This project is licensed under the MIT License.
See LICENSE for full terms.

---

## 🙌 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Streamlit](https://streamlit.io/) *(initial development stage before API conversion)*

---

## 📬 Contact

**Marwan Abdellah**  
📧 Email: [marawan.abdellah0@gmail.com](mailto:marawan.abdellah0@gmail.com)  
🔗 GitHub: [@MarwanAbdellah](https://github.com/MarwanAbdellah)
