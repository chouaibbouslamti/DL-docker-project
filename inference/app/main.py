from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models, transforms
from PIL import Image
import torch
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict, Any

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BLIP model for image captioning
processor = None
model_blip = None

def load_models():
    global processor, model_blip
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("BLIP model loaded successfully")

# Load models when the application starts
load_models()

# Charger le modèle ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Préparation images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Labels ImageNet
import json
import urllib.request
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as f:
    classes = [line.strip() for line in f]

@app.get("/")
def home():
    return {
        "message": "API inference ready",
        "endpoints": {
            "GET /": "API info",
            "POST /predict": "Image classification (ResNet18)",
            "POST /caption": "Generate image captions (BLIP)"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_t = preprocess(image)
        batch = torch.unsqueeze(img_t, 0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    try:
        with torch.no_grad():
            out = model(batch)
        _, indices = torch.sort(out, descending=True)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        return {
            "predictions": [
                {"label": classes[idx], "score": float(percentage[idx])}
                for idx in indices[0][:5]  # Top 5 predictions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/caption")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Read and process the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate caption
        inputs = processor(image, return_tensors="pt")
        
        # Generate caption
        out = model_blip.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Generate detailed caption with context
        inputs = processor(image, "a photography of", return_tensors="pt")
        out = model_blip.generate(**inputs, max_new_tokens=50)
        detailed_caption = processor.decode(out[0], skip_special_tokens=True)
        
        return {
            "simple_caption": caption,
            "detailed_caption": detailed_caption
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")
