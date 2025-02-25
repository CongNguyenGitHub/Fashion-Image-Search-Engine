import streamlit as st
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from demo_utils import (
    CustomVGG,
    CustomResNet50,
    CustomInceptionV3,
    find_top_k_images_vgg_faiss,
    find_top_k_images_resnet_faiss,
    find_top_k_images_inception_faiss,
    load_features,
)

# Load models
vgg_model = CustomVGG(num_classes=37)
resnet_model = CustomResNet50(num_classes=37)
inception_model = CustomInceptionV3(num_classes=37)

vgg_model.load_state_dict(
    torch.load("models/vgg_model/best_model.pt", map_location=torch.device("cpu"))
)
resnet_model.load_state_dict(
    torch.load("models/resnet_model/best_model.pt", map_location=torch.device("cpu"))
)
inception_model.load_state_dict(
    torch.load("models/inception_model/best_model.pt", map_location=torch.device("cpu"))
)

# Load features
vgg_features, vgg_labels, vgg_paths = load_features("features/cleaned_vgg_features.npz")
resnet_features, resnet_labels, resnet_paths = load_features(
    "features/cleaned_resnet_features.npz"
)
inception_features, inception_labels, inception_paths = load_features(
    "features/cleaned_inception_features.npz"
)

# App title
st.title("Image Similarity Search Demo")

# Sidebar for user inputs
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose a model:", ["VGG", "ResNet", "Inception"])
top_k = st.sidebar.slider(
    "Number of top-K images:", min_value=1, max_value=10, value=10
)

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Create two columns: left for uploaded image, right for similar images
    col1, col2 = st.columns([1, 2])  # Adjust the ratio as needed (e.g., 1:2)

    # Left column: Display uploaded image
    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    # Right column: Process and display similar images
    with col2:
        st.subheader(f"Top-{top_k} Similar Images")
        start_time = time.time()

        # Model selection
        if model_choice == "VGG":
            top_k_images, top_k_labels = find_top_k_images_vgg_faiss(
                pil_image, vgg_model, vgg_features, vgg_labels, vgg_paths, top_k
            )
        elif model_choice == "ResNet":
            top_k_images, top_k_labels = find_top_k_images_resnet_faiss(
                pil_image, resnet_model, resnet_features, resnet_labels, resnet_paths, top_k
            )
        elif model_choice == "Inception":
            top_k_images, top_k_labels = find_top_k_images_inception_faiss(
                pil_image,
                inception_model,
                inception_features,
                inception_labels,
                inception_paths,
                top_k,
            )

        end_time = time.time()


        # Display similar images in a grid
        num_cols = 3  # Number of images per row (adjust as needed)
        rows = (top_k + num_cols - 1) // num_cols  # Calculate total rows needed
        for row in range(rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                img_idx = row * num_cols + col_idx
                if img_idx < top_k:
                    cols[col_idx].image(top_k_images[img_idx], use_container_width=True)
                    cols[col_idx].write(f"Label: {top_k_labels[img_idx]}")