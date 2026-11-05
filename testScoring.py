
from PIL import Image
from script.calculateScore import calculate_similarity

masks = [
    "images/selection-mask1.png",
    "images/selection-mask2.png",
    "images/selection-mask3.png"
]

thresholds = [0.4, 0.5, 0.6, 0.7]

for mask_path in masks:
    user_image = Image.open(mask_path)

    print(f"\nResults for {mask_path}")

    for threshold in thresholds:
        score = calculate_similarity(
            user_image,
            threshold=threshold
        )

        print(f"Threshold {threshold}: {score:.2f}%")
