import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from granularity_vis import hyperbolic_norm

local_model_path = "your_image_path"
model = CLIPModel.from_pretrained(local_model_path, torch_dtype=torch.bfloat16).cuda().eval()
processor = CLIPProcessor.from_pretrained(local_model_path)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    vision_outputs = model.vision_model(**inputs)
    last_hidden_state = vision_outputs.last_hidden_state
    cls_token = last_hidden_state[:, 0, :] 
    features_tensor = last_hidden_state[:, 1:, :]  

if features_tensor.dim() == 3 and features_tensor.shape[0] == 1:
    features_tensor = features_tensor.squeeze(0) 

curvature = 0.01

h_dist, h_mean = hyperbolic_norm(features_tensor, curv=curvature)

print(f"\n=== Result (Curvature c={curvature}) ===")
print(f"Hyperbolic Norm Shape: {h_dist.shape}")
print(f"Mean Hyperbolic Norm: {h_mean.item():.4f}")