from transformers import pipeline
from transformers.image_utils import load_image
import torch
from granularity_vis import hyperbolic_norm
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)
feature_extractor = pipeline(
    model="your_image_path",
    task="image-feature-extraction", 
)
raw_features = feature_extractor(image)

features_tensor = torch.tensor(raw_features)

if features_tensor.dim() == 3 and features_tensor.shape[0] == 1:
    features_tensor = features_tensor.squeeze(0)

curvature = 0.01

h_dist, h_mean = hyperbolic_norm(features_tensor, curv=curvature)

print(f"\n=== Result (Curvature c={curvature}) ===")
print(f"Hyperbolic Norm Shape: {h_dist.shape}") 
print(f"Mean Hyperbolic Norm: {h_mean.item():.4f}")