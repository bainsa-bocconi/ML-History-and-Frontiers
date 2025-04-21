import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from lime import lime_image
from skimage.segmentation import mark_boundaries
from PIL import Image

"""
labels used in classification:
    "benign_keratosis-like_lesions": 0,
    "basal_cell_carcinoma": 1,
    "actinic_keratoses": 2,
    "vascular_lesions": 3,
    "melanocytic_Nevi": 4,
    "melanoma": 5,
    "dermatofibroma": 6
"""


model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=7)  # Adjust num_labels for your dataset
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"], np.array(image)


def predict(images):
    model.eval()
    inputs = [feature_extractor(images=Image.fromarray(img), return_tensors="pt")["pixel_values"] for img in images]
    inputs = torch.cat(inputs)  
    with torch.no_grad():
        outputs = model(inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

def explain_with_lime(image_path):
    
    pixel_values, original_image = preprocess_image(image_path) 
    
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        original_image, 
        predict, 
        top_labels=1,  # Only explain the top predicted label
        hide_color=0,  # Set the background color for masked parts
        num_samples=200  # Number of perturbations
    )

    # Get the explanation for the top predicted class
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=top_label, 
        positive_only=True, 
        num_features=5,  # Number of superpixels(features) to highlight
        hide_rest=False
    )

    explained_image = mark_boundaries(temp, mask)
    plt.figure(figsize=(10, 10))
    plt.imshow(explained_image)
    plt.title(f"Explanation for Label: {model.config.id2label.get(top_label, 'Unknown')}")
    plt.axis("off")
    plt.show()

# Example usage
image_path = r"C:\Users\Velina Todorova\Downloads\skin cancer_pic3.jpg"
explain_with_lime(image_path)
