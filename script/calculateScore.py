
import numpy as np
from PIL import Image


def calculate_similarity(user_image):
    # Load numerical Grad-CAM heatmap
    heatmap = np.load(
        "resnet_output/image_1_heatmap.npy"
    ).astype(np.float32)

    # Resize user's selection to match heatmap dimensions
    user_image = user_image.convert("L")
    user_image = user_image.resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.NEAREST
    )

    # Convert selection to binary values
    user_mask = (
        np.array(user_image) > 127
    ).astype(np.float32)

    if user_mask.shape != heatmap.shape:
        raise ValueError(
            f"Dimension mismatch: user mask {user_mask.shape}, "
            f"heatmap {heatmap.shape}"
        )

    # Calculate Soft Dice
    intersection = np.sum(user_mask * heatmap)

    denominator = (
        np.sum(user_mask) +
        np.sum(heatmap)
    )

    score = (
        2 * intersection / denominator
        if denominator > 0 else 0
    )

    # Return a percentage
    return float(score * 100)
