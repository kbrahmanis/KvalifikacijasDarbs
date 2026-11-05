/*
   1. Load image inside canvas
*/

const canvas = document.getElementById("imageCanvas");
const context = canvas.getContext("2d"); // gets access to <canvas> 2d drawing tools

// Make an invisible canvas layer 
const maskCanvas = document.createElement("canvas");
const maskContext = maskCanvas.getContext("2d");

const brushSize = 20;
let isDrawing = false;

// create image object
const image = new Image();

// assign its location
image.src = "images/000000004954.jpg"

// insert the image inside the canvas after loading
image.onload = () => {
    // Set visible canvas dimensions
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Set invisible mask dimensions
    maskCanvas.width = image.naturalWidth;
    maskCanvas.height = image.naturalHeight;

    // Draw original image
    context.drawImage(image, 0, 0);
}

/*
   2. Painting and selection inside the canvas
*/

// Convert mouse coordinates to image coordinates
function getPosition(event) {
    const rect = canvas.getBoundingClientRect();

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}


// Redraw the image and selection overlay
function renderCanvas() {
    // Clear visible canvas
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Draw original image
    context.drawImage(image, 0, 0);

    // Color the selected mask red
    context.save();

    context.fillStyle = "rgba(255, 0, 0, 0.4)";
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Keep red only where the mask is painted
    context.globalCompositeOperation = "destination-in";
    context.drawImage(maskCanvas, 0, 0);

    context.restore();

    // Draw the original image behind the overlay
    context.globalCompositeOperation = "destination-over";
    context.drawImage(image, 0, 0);
    context.globalCompositeOperation = "source-over";
}

// Draw a single circular brush point
function drawPoint(x, y) {
    maskContext.fillStyle = "white";

    maskContext.beginPath();
    maskContext.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    maskContext.fill();

    renderCanvas();
}

// Draw a continuous brush stroke
function drawLine(from, to) {
    maskContext.strokeStyle = "white";
    maskContext.lineWidth = brushSize;
    maskContext.lineCap = "round";
    maskContext.lineJoin = "round";

    maskContext.beginPath();
    maskContext.moveTo(from.x, from.y);
    maskContext.lineTo(to.x, to.y);
    maskContext.stroke();

    renderCanvas();
}

function autoFillShape() {
    if (strokePoints.length < 3) return;

    const first = strokePoints[0];
    const last = strokePoints[strokePoints.length - 1];

    // Distance between start and endpoint
    const distance = Math.hypot(
        last.x -first.x,
        last.y - first.y
    )

    const closingDistance = brushSize;
    if (distance > closingDistance) return;

    const path = new Path2D();
    path.moveTo(first.x, first.y);

    for (let i = 1; i < strokePoints.length; i++) {
        path.lineTo(strokePoints[i].x, strokePoints[i].y);
    }

    path.closePath();

    maskContext.fillStyle = "white";
    maskContext.fill(path);

    renderCanvas();
}

let lastPosition = null;

canvas.addEventListener("pointerdown", (event) => {
    if (!image.complete || !image.naturalWidth) return;

    isDrawing = true;
    canvas.setPointerCapture(event.pointerId);

    lastPosition = getPosition(event);

    strokePoints = [lastPosition];

    drawPoint(lastPosition.x, lastPosition.y);
});

canvas.addEventListener("pointermove", (event) => {
    if (!isDrawing) return;

    const currentPosition = getPosition(event);

    strokePoints.push(currentPosition);

    drawLine(lastPosition, currentPosition);
    lastPosition = currentPosition;
});

function stopDrawing() {
    if (!isDrawing) return;

    isDrawing = false;
    autoFillShape();

    lastPosition = null;
    strokePoints = [];
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("lostpointercapture", stopDrawing);

/*
    3. Buttons
*/

// Clear canvas

document.getElementById("clearButton").addEventListener("click", () => {
    // Remove all selected areas
    maskContext.clearRect(
        0, 0,
        maskCanvas.width,
        maskCanvas.height
    );

    // Redraw the original image
    renderCanvas();
});

// Submit users selection

document.getElementById("submitButton")
    .addEventListener("click", async () => {

        // Create export canvas
        const exportCanvas = document.createElement("canvas");

        exportCanvas.width = maskCanvas.width;
        exportCanvas.height = maskCanvas.height;

        const exportCtx = exportCanvas.getContext("2d");

        // Black background
        exportCtx.fillStyle = "black";
        exportCtx.fillRect(
            0, 0,
            exportCanvas.width,
            exportCanvas.height
        );

        // Add white selection
        exportCtx.drawImage(maskCanvas, 0, 0);

        // Convert mask to Base64 PNG
        const maskData = exportCanvas.toDataURL("image/png");

        try {
            // Send mask to Python backend
            const response = await fetch("/api/compare", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    mask: maskData
                })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(
                    result.error || "Comparison failed"
                );
            }

            // Display score
            console.log(
                `Similarity score: ${result.score}%`
            );

            document.getElementById("similarityScore")
                .textContent =
                `Similarity score: ${result.score}%`;

            // Display Grad-CAM overlay
            const heatmapImage =
                document.getElementById("imageHeatmap");

            heatmapImage.src =
                "./resnet_output/image_1_overlay.png";

            heatmapImage.hidden = false;

        } catch (error) {
            console.error("Error:", error);
        }
    });
