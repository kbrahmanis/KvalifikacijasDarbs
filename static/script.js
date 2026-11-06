
/* 
   1. CONFIGURATION
*/

let IMAGE_PATH = "./images/image_";
let HEATMAP_PATH = "./resnet_output/image_";
let IMAGE_ID = 1

const API_URL = "/api/compare";

const BRUSH_SIZE = 20;
const SELECTION_COLOR = "rgba(255, 0, 0, 0.4)";


/* 
   2. CANVAS AND ELEMENT SETUP
*/

// HTML elements
const canvas = document.getElementById("imageCanvas");
const context = canvas.getContext("2d");

const heatmapImage = document.getElementById("imageHeatmap");
const similarityScore = document.getElementById("similarityScore");
const aiPredictions = document.getElementById("results");

const clearButton = document.getElementById("clearButton");
const submitButton = document.getElementById("submitButton");
const nextButton = document.getElementById("nextButton");

// Invisible canvas for storing user selections
const maskCanvas = document.createElement("canvas");
const maskContext = maskCanvas.getContext("2d");

// Invisible canvas for rendering the red overlay
const overlayCanvas = document.createElement("canvas");
const overlayContext = overlayCanvas.getContext("2d");

// Original image
const image = new Image();

// Drawing state
let isDrawing = false;
let lastPosition = null;
let strokePoints = [];

// Hide results initially
heatmapImage.hidden = true;
similarityScore.textContent = "";
nextButton.hidden = true;
aiPredictions.hidden = true;

/* 
   3. IMAGE LOADING
*/

function loadImage() {
    image.onload = () => {
        const width = image.naturalWidth;
        const height = image.naturalHeight;

        // Match all canvas dimensions to the original image
        canvas.width = width;
        canvas.height = height;

        maskCanvas.width = width;
        maskCanvas.height = height;

        overlayCanvas.width = width;
        overlayCanvas.height = height;

        renderCanvas();
    };

    image.onerror = () => {
        console.error("Failed to load image:", IMAGE_PATH);
    };

    image.src = IMAGE_PATH + IMAGE_ID + ".jpg";
}


/* 
   4. CANVAS RENDERING
*/

// Redraw original image with the user's selection
function renderCanvas() {
    // Restore original image
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0);

    // Clear previous overlay
    overlayContext.clearRect(
        0, 0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    // Fill overlay with semi-transparent red
    overlayContext.fillStyle = SELECTION_COLOR;
    overlayContext.fillRect(
        0, 0,
        overlayCanvas.width,
        overlayCanvas.height
    );

    // Keep red only where the mask has been painted
    overlayContext.globalCompositeOperation = "destination-in";
    overlayContext.drawImage(maskCanvas, 0, 0);

    // Restore default drawing mode
    overlayContext.globalCompositeOperation = "source-over";

    // Draw selection over original image
    context.drawImage(overlayCanvas, 0, 0);
}


// Convert pointer coordinates to image coordinates
function getPosition(event) {
    const rect = canvas.getBoundingClientRect();

    return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height
    };
}


/* 
   5. DRAWING FUNCTIONS
*/

// Draw a single brush point
function drawPoint(x, y) {
    maskContext.fillStyle = "white";

    maskContext.beginPath();
    maskContext.arc(x, y, BRUSH_SIZE / 2, 0, Math.PI * 2);
    maskContext.fill();

    renderCanvas();
}


// Draw a continuous line between two points
function drawLine(from, to) {
    maskContext.strokeStyle = "white";
    maskContext.lineWidth = BRUSH_SIZE;
    maskContext.lineCap = "round";
    maskContext.lineJoin = "round";

    maskContext.beginPath();
    maskContext.moveTo(from.x, from.y);
    maskContext.lineTo(to.x, to.y);
    maskContext.stroke();

    renderCanvas();
}


// Automatically fill a closed shape
function autoFillShape() {
    if (strokePoints.length < 3) return;

    const first = strokePoints[0];
    const last = strokePoints[strokePoints.length - 1];

    // Check whether the stroke forms a closed shape
    const distance = Math.hypot(
        last.x - first.x,
        last.y - first.y
    );

    if (distance > BRUSH_SIZE) return;

    // Construct a path from the recorded points
    const path = new Path2D();
    path.moveTo(first.x, first.y);

    for (let i = 1; i < strokePoints.length; i++) {
        path.lineTo(strokePoints[i].x, strokePoints[i].y);
    }

    path.closePath();

    // Fill the enclosed area
    maskContext.fillStyle = "white";
    maskContext.fill(path);

    renderCanvas();
}


/* 
   6. POINTER EVENT HANDLERS
*/

function startDrawing(event) {
    if (!image.complete || !image.naturalWidth) return;

    isDrawing = true;
    canvas.setPointerCapture(event.pointerId);

    lastPosition = getPosition(event);
    strokePoints = [lastPosition];

    drawPoint(lastPosition.x, lastPosition.y);
}


function continueDrawing(event) {
    if (!isDrawing) return;

    const currentPosition = getPosition(event);

    strokePoints.push(currentPosition);

    drawLine(lastPosition, currentPosition);

    lastPosition = currentPosition;
}


function stopDrawing() {
    if (!isDrawing) return;

    isDrawing = false;

    autoFillShape();

    lastPosition = null;
    strokePoints = [];
}


/* 
   7. SELECTION MANAGEMENT
*/

// Clear the user's selection and previous results
function clearSelection() {
    maskContext.clearRect(
        0, 0,
        maskCanvas.width,
        maskCanvas.height
    );

    renderCanvas();

    // Hide previous results
    heatmapImage.hidden = true;
    similarityScore.textContent = "";
    aiPredictions.hidden = true;
}


// Convert selection into a black-and-white PNG
function exportMask() {
    const exportCanvas = document.createElement("canvas");

    exportCanvas.width = maskCanvas.width;
    exportCanvas.height = maskCanvas.height;

    const exportContext = exportCanvas.getContext("2d");

    // Black = unselected
    exportContext.fillStyle = "black";
    exportContext.fillRect(
        0, 0,
        exportCanvas.width,
        exportCanvas.height
    );

    // White = selected
    exportContext.drawImage(maskCanvas, 0, 0);

    return exportCanvas.toDataURL("image/png");
}


/* 
   8. BACKEND COMMUNICATION
*/

// Send user selection to Python and retrieve similarity score
async function calculateScore(maskData, image_id) {
    const response = await fetch(API_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            mask: maskData,
            id: image_id
        })
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(
            result.error || "Comparison failed"
        );
    }

    return result.score;
}

async function getPredictions() {
    const imageName = IMAGE_PATH.split("/").pop() + IMAGE_ID + ".jpg";

    const response = await fetch(
        `/api/predictions/${encodeURIComponent(imageName)}`
    );

    if (!response.ok) {
        throw new Error("Failed to load predictions");
    }

    return await response.json();
}

// Display the five predictions
function displayPredictions(predictions) {
    const container = document.getElementById("predictions");

    // Remove previous results
    container.replaceChildren();

    for (const prediction of predictions) {
        const item = document.createElement("p");

        item.textContent =
            `${prediction.category}: ${prediction.confidence.toFixed(2)}%`;

        container.appendChild(item);
    }

    aiPredictions.hidden = false;
}

// Display the similarity score and Grad-CAM overlay
function displayResults(score) {
    similarityScore.textContent =
        `Similarity score: ${score}%`;

    heatmapImage.src = HEATMAP_PATH + IMAGE_ID + "_overlay.png";
    heatmapImage.hidden = false;
}

// Handle submission
async function submitSelection() {
    if (!image.complete || !image.naturalWidth) return;

    submitButton.disabled = true;

    // Hide previous results while processing
    heatmapImage.hidden = true;
    similarityScore.textContent = "";
    aiPredictions.hidden = true;

    try {
        const maskData = exportMask();

        const score = await calculateScore(maskData, IMAGE_ID);
        const predictions = await getPredictions();

        displayResults(score);

        displayPredictions(predictions);

        nextButton.hidden = false;

    } catch (error) {
        console.error("Submission failed:", error);

        similarityScore.textContent =
            "Could not calculate similarity score.";

    } finally {
        submitButton.disabled = false;
    }
}

function getNextImage() {
    if (IMAGE_ID < 50) {
        IMAGE_ID++;
    } else {
        // To be implemented: A popup window for starting again
        IMAGE_ID = 0;
    }

    clearSelection();
    loadImage();
    nextButton.hidden = true;
    aiPredictions.hidden = true;
}


/* 
   9. EVENT LISTENERS
*/

// Canvas drawing
canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", continueDrawing);

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("lostpointercapture", stopDrawing);

// Buttons
clearButton.addEventListener("click", clearSelection);
submitButton.addEventListener("click", submitSelection);
nextButton.addEventListener("click", getNextImage);

/* 
   10. INITIALIZATION
*/

loadImage();
