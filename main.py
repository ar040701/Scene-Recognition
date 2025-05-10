from fastapi import FastAPI, UploadFile, File   # FastAPI imports
from fastapi.middleware.cors import CORSMiddleware   # CORSMiddleware imports  
from PIL import Image   # PIL imports
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import io
import timm

app = FastAPI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom label categories (40 classes)
class_labels = [
    "airport_terminal", "amphitheatre","amusement_park", "art_gallery",
    "bakery_shop","bar","bookstore", "botanical_garden","bridge",
    "bus interior","butchers shop","campsite","classroom","coffee_shop",
    "construction_site","courtyard","driveway","fire_station","fountain",
    "gas_station","harbour","highway","kindergarten_classroom","lobby",
    "market_outdoor","museum","office","parking_lot","phone_booth",
    "playground","railroad_track","restaurant","river","shed","staircase",
    "supermarket","swomming_pool_outdoor","track","valley","yard"
]

# Load your ViT model
# model = timm.create_model("vit-base-patch16-224", pretrained=False, num_classes=len(class_labels))
# from custom_vit import ViT

# model = ViT(
#     img_size=224,
#     patch_size=16,
#     num_classes=len(class_labels),
#     emb_size=768,
#     depth=12,
#     num_heads=12,
#     drop_p=0.1,
#     forward_drop_p=0.1
# )

from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')  # Assuming 40 classes)
state_dict = torch.load("vit_base.pth", map_location="cpu")
custom_classifier_weight = state_dict['classifier.weight']
custom_classifier_bias = state_dict['classifier.bias']
model.classifier = nn.Linear(model.classifier.in_features, len(class_labels))
model.classifier.weight.data = custom_classifier_weight
model.classifier.bias.data = custom_classifier_bias
model.load_state_dict(state_dict, strict=False)

model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(input_tensor).logits
            predicted_index = outputs.argmax(dim=1).item()
            predicted_label = class_labels[predicted_index]

        return {"class": predicted_label}

    except Exception as e:
        return {"error": str(e)}