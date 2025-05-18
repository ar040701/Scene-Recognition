from fastapi import FastAPI, UploadFile, File, Query   # FastAPI imports
from fastapi.middleware.cors import CORSMiddleware   # CORSMiddleware imports  
from PIL import Image   # PIL imports
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import io
import timm
import uvicorn
import os
import numpy as np
import matplotlib.pyplot as plt
import base64
from transformers import ViTForImageClassification, SwinForImageClassification

class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super(DinoClassifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.fc(x)
        return x

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

def get_attention_map(image_pil, model):
    # Convert image to tensor
    img_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, 224, 224]
    
    # Enable output attentions
    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        attentions = outputs.attentions  # Tuple of layer-wise attention: each is [1, heads, tokens, tokens]
    
    # Take the last layer attention
    attn = attentions[-1]  # [1, heads, tokens, tokens]
    attn = attn[0].mean(0)  # mean over heads => [tokens, tokens]
    
    # Attention from [CLS] to patches
    cls_attn = attn[0, 1:]  # from CLS token to all others (exclude CLS-to-CLS)
    cls_attn = cls_attn.reshape(14, 14).detach().cpu().numpy()  # ViT has 14x14 patches
    
    # Normalize and resize to image size
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    cls_attn_resized = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(image_pil.size, resample=Image.BILINEAR)
    cls_attn_resized = np.array(cls_attn_resized)

    # Convert to heatmap overlay
    heatmap = plt.cm.jet(cls_attn_resized / 255.0)[:, :, :3] * 255  # RGB heatmap
    heatmap = Image.fromarray(heatmap.astype(np.uint8)).convert("RGBA")
    image_pil = image_pil.convert("RGBA")
    overlay = Image.blend(image_pil, heatmap, alpha=0.5)

    # Convert to base64
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img

vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', attn_implementation="eager")
vit_model.classifier = nn.Linear(vit_model.classifier.in_features, len(class_labels))
vit_state = torch.load("vit_base.pth", map_location="cpu")
vit_model.classifier.weight.data = vit_state['classifier.weight']
vit_model.classifier.bias.data = vit_state['classifier.bias']
vit_model.load_state_dict(vit_state, strict=False)
vit_model.eval()

# dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# in_features=768
# dino_model.head = nn.Linear(in_features, len(class_labels))
# dino_state = torch.load("dinov2_80_calr_best.pth", map_location="cpu")
# dino_model.load_state_dict(dino_state, strict=False)
# dino_model.eval()

swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
swin_model.classifier = nn.Linear(swin_model.classifier.in_features, len(class_labels))
swin_state = torch.load("swin_66.pth", map_location="cpu")
swin_model.load_state_dict(swin_state, strict=False)
swin_model.eval()

print("VIT Classifier Weights Sum:", vit_model.classifier.weight.data.sum())
print("DINO Classifier Weights Sum:", dino_model.head.weight.data.sum())
print("Swin Classifier Weights Sum:", swin_model.classifier.weight.data.sum())

print(torch.allclose(vit_model.classifier.weight.data, dino_model.head.weight.data))
print(torch.allclose(vit_model.classifier.weight.data, swin_model.classifier.weight.data))

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_all_attention_maps(image_pil, model):
  try:
    img_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)
        attentions = outputs.attentions  # List of [1, heads, tokens, tokens]

    all_maps = {}
    image_pil = image_pil.convert("RGBA")

    for layer_idx, layer_attn in enumerate(attentions):
        for head_idx in range(layer_attn.shape[1]):
            attn = layer_attn[0, head_idx]  # [tokens, tokens]
            cls_attn = attn[0, 1:]  # CLS token to patch tokens
            cls_attn = cls_attn.reshape(14, 14).detach().cpu().numpy()

            # Normalize
            cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
            cls_attn_resized = Image.fromarray((cls_attn * 255).astype(np.uint8)).resize(
                image_pil.size, resample=Image.BILINEAR)
            cls_attn_resized = np.array(cls_attn_resized)

            # Heatmap overlay
            heatmap = plt.cm.jet(cls_attn_resized / 255.0)[:, :, :3] * 255
            heatmap = Image.fromarray(heatmap.astype(np.uint8)).convert("RGBA")
            overlay = Image.blend(image_pil, heatmap, alpha=0.5)

            # Convert to base64
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")

            key = f"layer_{layer_idx}_head_{head_idx}"
            all_maps[key] = f"data:image/png;base64,{base64_img}"

    return all_maps

  except Exception as e:
      return {"error": str(e)}
  
@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Query("vit", enum=["vit", "dino", "swin"])):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

        if model_name == "vit":
            model = vit_model
            with torch.no_grad():
                outputs = model(input_tensor, output_attentions=True)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            attention_maps = extract_all_attention_maps(image, model)

        elif model_name == "dino":
            model = dino_model
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            attention_maps = {}

        else:  # swin
            model = swin_model
            with torch.no_grad():
                outputs = model(input_tensor, output_attentions=True)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
            attention_maps = {}

        # Top-5 predictions
        top5_prob, top5_indices = torch.topk(probs, 5)
        top5 = [{"label": class_labels[idx], "score": round(prob.item(), 4)} for idx, prob in zip(top5_indices, top5_prob)]
        predicted_label = top5[0]["label"]

        return {
            "class": predicted_label,
            "top5": top5,
            "attention_map": attention_maps
        }


    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
     port = int(os.environ.get("PORT", 8000)) 
     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)