import cv2
import torch
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# Load the pre-trained ViT model from Hugging Face
model_name = "google/vit-base-patch16-224-in21k"
# model_name = "apple/mobilevit-small"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load and preprocess the image
image_path = "img/image.png"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Convert to grayscale and apply thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Convert grayscale to RGB
pil_image = Image.fromarray(binary).convert("RGB")

# Preprocess the image for ViT
inputs = feature_extractor(images=pil_image, return_tensors="pt")

# Predict the shape
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits
    predicted_class = predictions.argmax(-1).item()

# Define custom labels for shape classification
shape_labels = {0: "Circle", 1: "Triangle", 2: "Square", 3: "Rectangle", 4: "Other"}

# Get the predicted shape name
shape_name = shape_labels.get(predicted_class, "Unknown")

# Display the result
cv2.putText(image, f"Detected Shape: {shape_name}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
