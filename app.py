
from flask import Flask, request, jsonify, send_from_directory
from script.calculateScore import calculate_similarity

from PIL import Image
from io import BytesIO
import base64
import binascii

import json
import re
from pathlib import Path

app = Flask(__name__)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parent
    / "resnet_output"
    / "predictions.json"
)

@app.route("/api/compare", methods=["POST"])
def compare():
    try:
        # Get JSON sent by JavaScript
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400

        mask_data = data.get("mask")

        if not isinstance(mask_data, str):
            return jsonify({"error": "Missing mask"}), 400

        # Remove Base64 data URL prefix
        prefix = "data:image/png;base64,"

        if not mask_data.startswith(prefix):
            return jsonify({"error": "Invalid image format"}), 400

        encoded_image = mask_data[len(prefix):]

        # Convert Base64 string into image bytes
        image_bytes = base64.b64decode(
            encoded_image,
            validate=True
        )

        # Open image using PIL
        user_image = Image.open(BytesIO(image_bytes))
        user_image.load()

        user_image_id = data.get("id")

        # Call our scoring function
        score = calculate_similarity(user_image, user_image_id)

        # Return result to JavaScript
        return jsonify({
            "score": round(score, 2)
        })

    except (ValueError, binascii.Error, OSError):
        return jsonify({
            "error": "Could not process selection"
        }), 400

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory("images", filename)


@app.route("/resnet_output/<path:filename>")
def outputs(filename):
    # Only expose the generated overlay images
    if not filename.endswith("_overlay.png"):
        return jsonify({"error": "Not found"}), 404

    return send_from_directory("resnet_output", filename)


@app.route("/api/predictions/<image_name>", methods=["GET"])
def get_image_predictions(image_name):
    # Only allow filenames like image_1.jpg
    if not re.fullmatch(r"image_[1-9]\d*\.jpg", image_name):
        return jsonify({"error": "Invalid image name"}), 400

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as file:
        predictions = json.load(file)

    image_predictions = predictions.get(image_name)

    if image_predictions is None:
        return jsonify({"error": "Predictions not found"}), 404

    return jsonify(image_predictions)

if __name__ == "__main__":
    app.run(debug=True)
