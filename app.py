from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import os
from ultralytics import YOLO

app = Flask(__name__)
# Load model from a relative path or environment variable
model_path = os.environ.get('MODEL_PATH', 'train_yolo9_v1/weights/best.pt')
model = YOLO(model_path)
latest_detections = []

def decide_satellite_movement(objects):
    if not objects:
        return "No objects detected, stay in position"

    # Simple decision: Move opposite to the closest object's horizontal position
    # Prioritize objects in the "left" or "right" to avoid collision
    for obj in objects:
        if "left" in obj['position']:
            return "Move satellite right"
        elif "right" in obj['position']:
            return "Move satellite left"

    # If no objects are on left or right, check if any are in center
    for obj in objects:
        if obj['position'] == "center":
            # Could refine further based on vertical position or size
            return "Move satellite slightly right"  # Arbitrary choice, can adjust logic

    # Default: No critical objects in horizontal extremes or center
    return "Stay in position"

@app.route('/predict', methods=['POST'])
def predict():
    global latest_detections
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model(image_cv)
    detections = []
    img_height, img_width = image_cv.shape[:2]

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            conf = float(box.conf)
            cls = int(box.cls)
            label = model.names[cls]

            # Calculate center point of object
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Determine horizontal direction
            if center_x < img_width / 3:
                horizontal_dir = "left"
            elif center_x > 2 * img_width / 3:
                horizontal_dir = "right"
            else:
                horizontal_dir = "center"

            # Determine vertical direction
            if center_y < img_height / 3:
                vertical_dir = "top"
            elif center_y > 2 * img_height / 3:
                vertical_dir = "bottom"
            else:
                vertical_dir = "middle"

            # Combine directions into position
            if horizontal_dir == "center" and vertical_dir == "middle":
                position = "center"
            else:
                position = f"{vertical_dir}-{horizontal_dir}"

            detections.append({
                "class": label,
                "class_id": cls,
                "width": width,
                "height": height,
                "center_x": center_x,
                "center_y": center_y,
                "position": position,
                "confidence": conf
            })

    latest_detections = detections
    movement = decide_satellite_movement(detections)
    return jsonify({"detections": detections, "movement": movement})

@app.route('/results', methods=['GET'])
def get_results():
    movement = decide_satellite_movement(latest_detections)
    return jsonify({"detections": latest_detections, "movement": movement})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)