import numpy as np
from PIL import Image

def calculate_similarity(user_image, threshold=0.4):
    # Load numerical Grad-CAM heatmap
    heatmap = np.load(
        "resnet_output/image_1_heatmap.npy"
    ).astype(np.float32)

    # Normalize heatmap to [0,1]
    heatmap = np.clip(heatmap, 0, None)

    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value
    
    # Resize user's selection to match heatmap dimensions
    user_image = user_image.convert("L")
    user_image = user_image.resize(
        (heatmap.shape[1], heatmap.shape[0]),
        Image.Resampling.NEAREST
    )

    # Convert selection to binary mask
    user_mask = np.array(user_image) > 127

    # Convert grad-Cam into binart mask
    ai_mask = heatmap >= threshold

    # Calculate Dice
    intersection = np.logical_and(user_mask, ai_mask).sum()

    user_area = user_mask.sum()
    ai_area = ai_mask.sum()

    denominator = user_area + ai_area

    score = (
        2 * intersection / denominator
        if denominator > 0 else 0
    )

    # 7. Debug information
    print(f"User selected: {user_area / user_mask.size * 100:.2f}%")
    print(f"AI selected: {ai_area / ai_mask.size * 100:.2f}%")
    print(f"Dice score: {score * 100:.2f}%")

    # Return a percentage
    return float(score * 100)
