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

IMAGE_URLS = [
    "http://images.cocodataset.org/test-stuff2017/000000004954.jpg",
    "http://images.cocodataset.org/test-stuff2017/000000024309.jpg",
    "http://images.cocodataset.org/test-stuff2017/000000028117.jpg",
    "http://images.cocodataset.org/test-stuff2017/000000006149.jpg",
]

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
def load_images(urls):

    images = []
    tensors = []

    for url in urls:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        images.append(image)
        tensors.append(image_transform(image))

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

"""Print the top 5 classifications for an image."""
def print_predictions(probabilities, labels, image_url):

    top_probs, top_indices = torch.topk(probabilities, 5)

    print(f"\nTop 5 predictions for {image_url}:")

    for rank, (prob, class_idx) in enumerate(
        zip(top_probs, top_indices), start=1
    ):
        class_name = labels[class_idx.item()]
        confidence = prob.item() * 100

        print(f"  {rank}. {class_name}: {confidence:.2f}%")

    print("-" * 40)


"""
Process each image's results.

Saves:
    - Numerical heatmap (.npy)
    - Colored overlay (.png)

Also prints the top 5 class predictions.
"""
def save_results(images, heatmaps, probabilities, labels, urls):

    for i, original_image in enumerate(images):

        image_number = i + 1

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
        print_predictions(
            probabilities[i],
            labels,
            urls[i]
        )

        print(
            f"Image {image_number} saved: "
            f"{original_image.width}x{original_image.height}"
        )

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
            IMAGE_URLS
        )

    finally:
        # Always remove the hook when finished.
        hook.remove()

    print("\nAll Grad-CAM files saved successfully!")


if __name__ == "__main__":
    main()


# def restore_heatmap(heatmap, original_image, resize_size=256, crop_size=224):
#     # Original image dimensions
#     original_width, original_height = original_image.size

#     # 1. Recreate the dimensions after Resize()
#     if original_width < original_height:
#         resized_width = resize_size
#         resized_height = int(
#             resize_size * original_height / original_width
#         )
#     else:
#         resized_height = resize_size
#         resized_width = int(
#             resize_size * original_width / original_height
#         )

#     # 2. Resize Grad-CAM to the model's crop dimensions
#     heatmap_image = Image.fromarray(
#         heatmap.astype(np.float32)
#     )

#     heatmap_image = heatmap_image.resize(
#         (crop_size, crop_size),
#         Image.Resampling.BILINEAR
#     )

#     heatmap_resized = np.asarray(heatmap_image)

#     # 3. Create an empty heatmap for the resized image
#     full_heatmap = np.zeros(
#         (resized_height, resized_width),
#         dtype=np.float32
#     )

#     # 4. Calculate where CenterCrop() extracted the image
#     left = int(round((resized_width - crop_size) / 2.0))
#     top = int(round((resized_height - crop_size) / 2.0))

#     # 5. Place the Grad-CAM heatmap back into that region
#     full_heatmap[
#         top:top + crop_size,
#         left:left + crop_size
#     ] = heatmap_resized

#     # 6. Resize to original image dimensions
#     full_heatmap_image = Image.fromarray(full_heatmap)

#     full_heatmap_image = full_heatmap_image.resize(
#         (original_width, original_height),
#         Image.Resampling.BILINEAR
#     )

#     return np.asarray(full_heatmap_image, dtype=np.float32)


# # 1. Load ResNet50 using timm
# resnet50 = timm.create_model('resnet50', pretrained=True)
# resnet50.eval().to(device)

# # Store GradCam intermediate activations

# activations = None

# def save_activations(module, inputs, output):
#     global activations
#     activations = output

# target_layer = resnet50.layer4[-1]
# hook = target_layer.register_forward_hook(save_activations)
# print("Grad-Cam hook registered")

# # 2. Define standard ImageNet preprocessing
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], 
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# # 3. Fetch ImageNet class labels mapping (for printing readable names)
# LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# imagenet_labels = json.loads(requests.get(LABELS_URL).text)

# uris = [
#     "http://images.cocodataset.org/test-stuff2017/000000004954.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000024309.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000028117.jpg",
#     "http://images.cocodataset.org/test-stuff2017/000000006149.jpg",
# ]

# # 4. Prepare batch manually using standard PIL and torchvision transforms
# tensor_list = []
# images_list = []

# for uri in uris:
#     img = Image.open(requests.get(uri, stream=True).raw).convert("RGB")
#     images_list.append(img)
#     tensor_list.append(transform(img))

# batch = torch.stack(tensor_list).to(device)

# # 5. Inference
# logits = resnet50(batch)

# output = torch.nn.functional.softmax(logits, dim=1)

# print("Logits shape:", logits.shape)
# print("Activations shape:", activations.shape)

# # 6. Get top-5 predictions manually (replaces utils.pick_n_best)
# top5_prob, top5_catid = torch.topk(output, 5, dim=1)

# # 6.1. Grad-Cam: select the highest scoring class for each image
# target_classes = top5_catid[:,0]

# # 6.2 Grad-Cam: get corresponding raw logit from each image
# batch_indices = torch.arange(len(batch), device=device)

# selected_logits = logits[
#     batch_indices,
#     target_classes
# ]

# # 6.3 Grad-Cam: calculate gradients of selected logits
# gradients = torch.autograd.grad(
#     outputs=selected_logits.sum(),
#     inputs=activations
# )[0]

# print("Gradient shape:", gradients.shape)

# # 7. GRAD-CAM: Generate one heatmap per image

# # Average gradients across spatial dimensions
# weights = gradients.mean(dim=(2, 3), keepdim=True)

# # Multiply activations by their importance weights
# cam = (weights * activations).sum(dim=1)

# # Remove negative contributions
# cam = torch.relu(cam)

# print("CAM shape:", cam.shape)

# # 8. GRAD-CAM: Resize heatmaps

# # Add channel dimension: [4, 1, 7, 7]
# cam = cam.unsqueeze(1)

# # Resize to 224 x 224
# cam = F.interpolate(
#     cam,
#     size=(224, 224),
#     mode="bilinear",
#     align_corners=False
# )

# # Remove channel dimension: [4, 224, 224]
# cam = cam.squeeze(1)

# # Convert to NumPy
# cam = cam.detach().cpu().numpy()

# # Normalize each heatmap independently
# for i in range(len(cam)):
#     cam[i] = cam[i] - cam[i].min()

#     if cam[i].max() > 0:
#         cam[i] = cam[i] / cam[i].max()

# print("Final heatmap shape:", cam.shape)

# # 8. Display predictions and save Grad-CAM images

# # test this
# display_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
# ])

# for i, uri in enumerate(uris):

#     # 1. Get original image
#     original_image = images_list[i].convert("RGB")

#     # 2. Get numerical Grad-CAM heatmap
#     heatmap = np.asarray(cam[i], dtype=np.float32)

#     # 3. Restore heatmap to original image coordinates
#     heatmap = restore_heatmap(
#         heatmap,
#         original_image,
#         resize_size=256,
#         crop_size=224
#     )

#     # 4. Save aligned numerical heatmap
#     np.save(
#         f"{output_dir}/image_{i + 1}_heatmap.npy",
#         heatmap
#     )

#     # 5. Convert original image to NumPy
#     original = np.asarray(
#         original_image
#     ).astype(np.float32) / 255.0

#     # 6. Convert heatmap to RGB
#     heatmap_rgb = plt.get_cmap("jet")(heatmap)[..., :3]

#     # 7. Calculate transparency
#     alpha = heatmap[..., None] * 0.65

#     # 8. Blend original image and heatmap
#     overlay = original * (1 - alpha) + heatmap_rgb * alpha

#     # 9. Save overlay at original resolution
#     overlay_image = Image.fromarray(
#         (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
#     )

#     overlay_image.save(
#         f"{output_dir}/image_{i + 1}_overlay.png"
#     )

#     # 10. Print classification results
#     print(f"\nTop 5 predictions for {uri}:")

#     for j in range(5):
#         class_idx = top5_catid[i][j].item()
#         class_name = imagenet_labels[class_idx]
#         probability = top5_prob[i][j].item() * 100

#         print(f"  {j + 1}. {class_name}: {probability:.2f}%")

#     print("-" * 40)

# hook.remove()

# print("All Grad-CAM files saved successfully!")




