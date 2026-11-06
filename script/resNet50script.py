import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
import timm

from PIL import Image
from matplotlib import colormaps
from torchvision import transforms

# --- 1. CONFIGURATION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("resnet_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_INPUT_SIZE = 224
OVERLAY_OPACITY = 0.65

LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/"
    "imagenet-simple-labels/master/imagenet-simple-labels.json"
)

# IMAGE_URLS = [
#     "http://images.cocodataset.org/test-stuff2017/000000004954.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000024309.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000028117.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000006149.jpg",
# ]

IMAGE_DIR = Path("images")

IMAGE_URLS = sorted(
    IMAGE_DIR.glob("image_*.jpg"),
    key=lambda path: int(path.stem.split("_")[-1])
)

# Resize the ENTIRE image to the model's input dimensions.
# No CenterCrop means no image regions are discarded.
image_transform = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- 2. MODEL SETUP

"""Load a pretrained ResNet50 model."""
def load_model():
    
    model = timm.create_model("resnet50", pretrained=True)
    model.eval()
    model.to(DEVICE)

    return model

"""
Register a forward hook on the last ResNet block.

The hook stores activations needed later for 
calculating Grad-CAM.
"""
def register_gradcam_hook(model):

    saved = {}

    def save_activations(module, inputs, output):
        saved["activations"] = output

    target_layer = model.layer4[-1]
    hook = target_layer.register_forward_hook(save_activations)

    return saved, hook

# --- 3. IMAGE PREP

"""
Download images and prepare them for the model.

Returns:
    images: Original PIL images
    batch:  Preprocessed PyTorch tensor
"""
# def load_images(urls):

#     images = []
#     tensors = []

#     for url in urls:
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()

#         image = Image.open(BytesIO(response.content)).convert("RGB")

#         images.append(image)
#         tensors.append(image_transform(image))

#     batch = torch.stack(tensors).to(DEVICE)

#     return images, batch

def load_images(image_paths):
    images = []
    tensors = []

    for path in image_paths:
        # Load image directly from local storage
        with Image.open(path) as img:
            image = img.convert("RGB")

        # Store original image
        images.append(image)

        # Apply ResNet50 preprocessing
        tensors.append(image_transform(image))

    # Combine processed images into a batch
    batch = torch.stack(tensors).to(DEVICE)

    return images, batch


"""Download readable names for ImageNet classes."""
def load_labels():

    response = requests.get(LABELS_URL, timeout=30)
    response.raise_for_status()

    return json.loads(response.text)

# --- 4. INFERENCE AND GRAD-CAM

"""
Run classification and generate one Grad-CAM heatmap
for each image in the batch.

Grad-CAM is calculated for each image's
highest-scoring predicted class.

Returns:
    probabilities: Class probabilities
    heatmaps:      Normalized heatmaps [N, 224, 224]
"""
def generate_gradcam(model, batch, saved):

    # Run the model
    logits = model(batch)

    probabilities = F.softmax(logits, dim=1)

    # Select the best class for each image
    target_classes = logits.argmax(dim=1)

    batch_indices = torch.arange(
        batch.shape[0],
        device=DEVICE
    )

    selected_logits = logits[batch_indices, target_classes]

    # Calculate gradients
    activations = saved["activations"]

    gradients = torch.autograd.grad(
        outputs=selected_logits.sum(),
        inputs=activations
    )[0]

    # Calculate Grad-CAM weights
    # Average the gradients across spatial dimensions.
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Generate heatmaps
    cam = (weights * activations).sum(dim=1)

    # Remove negative contributions.
    cam = F.relu(cam)

    # Resize heatmaps to model input dimensions
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)

    # Normalize each heatmap to [0, 1]
    cam_min = cam.amin(dim=(1, 2), keepdim=True)
    cam_max = cam.amax(dim=(1, 2), keepdim=True)

    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmaps = cam.detach().cpu().numpy()

    return probabilities.detach(), heatmaps

# --- 5. OUTPUT 

"""
Resize the Grad-CAM heatmap to the original image size.

Because the entire image was resized during preprocessing,
the heatmap covers the entire original image.
"""
def resize_heatmap(heatmap, original_image):

    original_width, original_height = original_image.size

    heatmap_image = Image.fromarray(
        heatmap.astype(np.float32)
    )

    heatmap_image = heatmap_image.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR
    )

    return np.asarray(heatmap_image, dtype=np.float32)

"""
Blend the original image with the colored Grad-CAM heatmap.

Higher importance = stronger heatmap visibility.
Lower importance = original image remains visible.
"""
def create_overlay(original_image, heatmap):

    # Convert original image to values between 0 and 1.
    original = np.asarray(
        original_image.convert("RGB"),
        dtype=np.float32
    ) / 255.0

    # Map heatmap values to RGB colors.
    heatmap_rgb = colormaps["jet"](heatmap)[..., :3]

    # Transparency depends on importance.
    alpha = heatmap[..., None] * OVERLAY_OPACITY

    # Blend original and colored heatmap.
    overlay = original * (1 - alpha) + heatmap_rgb * alpha

    # Convert back to a PIL image.
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(overlay)

"""Get the top 5 classifications for an image."""

def get_predictions(probabilities, labels):
    top_probs, top_indices = torch.topk(probabilities, 5)

    predictions = []

    for prob, class_idx in zip(top_probs, top_indices):
        predictions.append({
            "category": labels[class_idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return predictions


"""
Process each image's results.

Saves:
    - Numerical heatmap (.npy)
    - Colored overlay (.png)

Also prints the top 5 class predictions.
"""
def save_results(images, heatmaps, probabilities, labels):

    all_predictions = {}

    for i, original_image in enumerate(images):

        image_number = i + 1
        image_name = f"image_{image_number}.jpg"

        # Restore heatmap to ORIGINAL image dimensions.
        heatmap = resize_heatmap(
            heatmaps[i],
            original_image
        )

        # Save numerical heatmap.
        np.save(
            OUTPUT_DIR / f"image_{image_number}_heatmap.npy",
            heatmap
        )

        # Create and save the overlay.
        overlay_image = create_overlay(original_image, heatmap)

        overlay_image.save(
            OUTPUT_DIR / f"image_{image_number}_overlay.png"
        )

        # Print classification results.
        all_predictions[image_name] = get_predictions(
            probabilities[i],
            labels
        )

        print(
            f"Image {image_number} saved: "
            f"{original_image.width}x{original_image.height}"
        )

    with open(
        OUTPUT_DIR / "predictions.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_predictions,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("All predictions saved successfully!")

# --- 6. MAIN

def main():

    print(f"Using {DEVICE} for inference")

    # Load the AI model and labels.
    print("Loading model...")
    model = load_model()
    labels = load_labels()

    # Register the Grad-CAM hook.
    saved, hook = register_gradcam_hook(model)

    try:
        # Download images and prepare the input batch.
        print("Preparing images...")
        images, batch = load_images(IMAGE_URLS)

        # Run inference and generate Grad-CAM heatmaps.
        print("Generating Grad-CAM heatmaps...")
        probabilities, heatmaps = generate_gradcam(
            model,
            batch,
            saved
        )

        print("Heatmap batch shape:", heatmaps.shape)

        # Save overlays and numerical heatmaps.
        print("Saving results...")
        save_results(
            images,
            heatmaps,
            probabilities,
            labels,
        )

    finally:
        # Always remove the hook when finished.
        hook.remove()

    print("\nAll Grad-CAM files saved successfully!")


if __name__ == "__main__":
    main()




