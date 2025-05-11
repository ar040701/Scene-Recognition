from fastapi import FastAPI, UploadFile, File   # FastAPI imports
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
from transformers import ViTForImageClassification

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


model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', attn_implementation="eager")  # Assuming 40 classes)
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
            outputs = model(input_tensor,output_attentions=True)
            logits=outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]

            # Get top-5 predictions
            top5_prob, top5_indices = torch.topk(probs, 5)
            top5 = [
                {"label": class_labels[idx], "score": round(prob.item(), 4)}
                for idx, prob in zip(top5_indices, top5_prob)
            ]

            predicted_label = top5[0]["label"]  # top-1 class
            # predicted_index = logits.argmax(dim=1).item()
            # predicted_label = class_labels[predicted_index]
        
        attention_map_base64 = get_attention_map(image, model)

        return {"class": predicted_label,
                "top5": top5,
                "attention_map": f"data:image/png;base64,{attention_map_base64}"}

    except Exception as e:
        return {"error": str(e)}
    
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000)) 
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)