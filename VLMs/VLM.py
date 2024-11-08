"""
Vision-Language Matching System using CLIP

This script implements an image-caption matching system using OpenAI's CLIP model.
It processes a collection of images and predefined captions, computing their 
embeddings to find the most suitable caption for each image based on semantic similarity.
The system utilizes the ViT-B-32 architecture pretrained by OpenAI to encode both
images and text into the same feature space, enabling direct comparison through
cosine similarity. The best matching caption for each image is displayed alongside 
the image itself.

Authors:
    - Arsenio Hugo
    - Mello Grand Matteo
    - Bertamini Riccardo

Requirements:
    - open_clip_torch
    - PyTorch
    - Pillow (PIL)
    - matplotlib

The script will:
    1. Load the pretrained CLIP ViT-B-32 model
    2. Process a collection of images and captions
    3. Compute embeddings for both images and text
    4. Calculate similarity scores between all pairs
    5. Display each image with its best matching caption

Returns:
    None. Displays images with their matched captions using matplotlib.
"""

import open_clip
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device='cpu')

image_folder = Path("/Users/hugoarsenio1/Desktop/YOLO/ProjectVLM/images")
image_paths = list(image_folder.glob("*.png"))
images = [Image.open(image_path) for image_path in image_paths]

captions = [
    "When you realize it’s Monday.",
    "Me trying to act normal in public.",
    "That face you make when someone says 'be yourself'.",
    "My reaction when there’s no Wi-Fi.",
    "Current mood: confused.",
    "Current mood: thriving on coffee and chaos.",
    "When you hear someone say 'free food.'",
    "Me trying to act professional on a video call.",
    "When you accidentally open the front camera."
]

tokenizer = open_clip.get_tokenizer('ViT-B-32')
text_inputs = torch.cat([tokenizer(caption) for caption in captions]).to("cpu")

image_features = []
for image in images:
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image_input)
        image_features.append(features)

if image_features:
    image_features = torch.cat(image_features)
else:
    raise ValueError("No images were processed successfully")

with torch.no_grad():
    text_features = model.encode_text(text_inputs)

    similarities = image_features @ text_features.T
    best_matches = similarities.argmax(dim=1)

for i, image in enumerate(images):
    plt.imshow(image)
    plt.axis('off')
    plt.title(captions[best_matches[i]], fontsize=14)
    plt.show()