import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import timm
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import json
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
output_dir = "resnet_output"
os.makedirs(output_dir, exist_ok=True)

print(f'Using {device} for inference')

# 1. Load ResNet50 using timm
resnet50 = timm.create_model('resnet50', pretrained=True)
resnet50.eval().to(device)

# Store GradCam intermediate activations

activations = None

def save_activations(module, inputs, output):
    global activations
    activations = output

target_layer = resnet50.layer4[-1]
hook = target_layer.register_forward_hook(save_activations)
print("Grad-Cam hook registered")

# 2. Define standard ImageNet preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 3. Fetch ImageNet class labels mapping (for printing readable names)
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
imagenet_labels = json.loads(requests.get(LABELS_URL).text)

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

# 4. Prepare batch manually using standard PIL and torchvision transforms
tensor_list = []
images_list = []

for uri in uris:
    img = Image.open(requests.get(uri, stream=True).raw).convert("RGB")
    images_list.append(img)
    tensor_list.append(transform(img))

batch = torch.stack(tensor_list).to(device)

# 5. Inference
logits = resnet50(batch)

output = torch.nn.functional.softmax(logits, dim=1)

print("Logits shape:", logits.shape)
print("Activations shape:", activations.shape)

# 6. Get top-5 predictions manually (replaces utils.pick_n_best)
top5_prob, top5_catid = torch.topk(output, 5, dim=1)

# 6.1. Grad-Cam: select the highest scoring class for each image
target_classes = top5_catid[:,0]

# 6.2 Grad-Cam: get corresponding raw logit from each image
batch_indices = torch.arange(len(batch), device=device)

selected_logits = logits[
    batch_indices,
    target_classes
]

# 6.3 Grad-Cam: calculate gradients of selected logits
gradients = torch.autograd.grad(
    outputs=selected_logits.sum(),
    inputs=activations
)[0]

print("Gradient shape:", gradients.shape)

# 7. GRAD-CAM: Generate one heatmap per image

# Average gradients across spatial dimensions
weights = gradients.mean(dim=(2, 3), keepdim=True)

# Multiply activations by their importance weights
cam = (weights * activations).sum(dim=1)

# Remove negative contributions
cam = torch.relu(cam)

print("CAM shape:", cam.shape)

# 8. GRAD-CAM: Resize heatmaps

# Add channel dimension: [4, 1, 7, 7]
cam = cam.unsqueeze(1)

# Resize to 224 x 224
cam = F.interpolate(
    cam,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# Remove channel dimension: [4, 224, 224]
cam = cam.squeeze(1)

# Convert to NumPy
cam = cam.detach().cpu().numpy()

# Normalize each heatmap independently
for i in range(len(cam)):
    cam[i] = cam[i] - cam[i].min()

    if cam[i].max() > 0:
        cam[i] = cam[i] / cam[i].max()

print("Final heatmap shape:", cam.shape)

# 8. Display predictions and save Grad-CAM images

# test this
display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

for i, uri in enumerate(uris):

    # Reproduce the same resize and crop used by the model
    # display_image = transform[0](images_list[i])
    # display_image = transform[1](display_image)
    display_image = display_transform(images_list[i])

    # Get the Grad-CAM heatmap
    heatmap = cam[i]

    # Create a figure with original, heatmap, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original model input
    axes[0].imshow(display_image)
    axes[0].set_title("Model input")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(display_image)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=heatmap * 0.65,
        vmin=0,
        vmax=1
    )
    axes[2].set_title(
        f"Prediction: {imagenet_labels[target_classes[i].item()]}"
    )
    axes[2].axis("off")

    plt.tight_layout()

    # Save comparison
    plt.savefig(
        f"{output_dir}/image_{i + 1}_gradcam.png",
        dpi=150
    )

    # Save heatmap as a separate image
    plt.imsave(
        f"{output_dir}/image_{i + 1}_heatmap.png",
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    # Save numerical heatmap for future scoring
    np.save(
        f"{output_dir}/image_{i + 1}_heatmap.npy",
        heatmap
    )

    # Print classification results
    print(f"\nTop 5 predictions for {uri}:")

    for j in range(5):
        class_idx = top5_catid[i][j].item()
        class_name = imagenet_labels[class_idx]
        probability = top5_prob[i][j].item() * 100

        print(f"  {j + 1}. {class_name}: {probability:.2f}%")

    plt.show()
    plt.close(fig)

# Remove hook after finishing
hook.remove()

print("All Grad-CAM images saved successfully!")

# 9.

for i, uri in enumerate(uris):
    # img = images_list[i]
    # img.thumbnail((256, 256), Image.LANCZOS)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    
    print(f"Top 5 predictions for {uri}:")
    for j in range(5):
        class_idx = top5_catid[i][j].item()
        class_name = imagenet_labels[class_idx]
        probability = top5_prob[i][j].item() * 100
        print(f"  {j+1}. {class_name}: {probability:.2f}%")
    print("-" * 40)


