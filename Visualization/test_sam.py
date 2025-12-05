import torch
from transformers import SamModel, SamProcessor
from transformers.image_utils import load_image
from granularity_vis import hyperbolic_norm

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "your_image_path"
model = SamModel.from_pretrained(model_id).to(device).eval()
processor = SamProcessor.from_pretrained(model_id)

raw_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")

inputs = processor(raw_image, return_tensors="pt").to(device)

with torch.no_grad():
    vision_outputs = model.vision_encoder(pixel_values=inputs.pixel_values)
    image_embeddings = vision_outputs.last_hidden_state


features_tensor = image_embeddings.permute(0, 2, 3, 1).reshape(-1, 256)

if features_tensor.dim() == 3 and features_tensor.shape[0] == 1:
    features_tensor = features_tensor.squeeze(0)
    
curvature = 0.01 

h_dist, h_mean = hyperbolic_norm(features_tensor, curv=curvature)

print(f"\n=== Result (Curvature c={curvature}) ===")
print(f"Hyperbolic Norm Shape: {h_dist.shape}")
print(f"Mean Hyperbolic Norm: {h_mean.item():.4f}")