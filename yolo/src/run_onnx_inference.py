#run as "python run_onnx_inference.py <onnx file with full path> <image with full path>"
import onnxruntime as ort
import cv2
import numpy as np
import torch
import sys


# Load the ONNX model
onnx_model_path = "sys.argv[1]"
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider']) # Use GPU if available

# Load and preprocess the image
image_path = "sys.argv[2]" # Replace with your image path
image = cv2.imread(image_path)
original_h, original_w = image.shape[:2]

# Preprocessing (adjust according to your model's input requirements)
input_size = 640 # Assuming a 640x640 input for YOLO
img_resized = cv2.resize(image, (input_size, input_size))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0
img_transposed = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
img_input = np.expand_dims(img_transposed, axis=0) # Add batch dimension

# Run inference
input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, {input_name: img_input})


# 4. Access and print outputs
print("Model Outputs:")
for i, output_tensor in enumerate(outputs):
    output_name = session.get_outputs()[i].name
    print(f"Output {i} (Name: {output_name}):")
    print(f"  Shape: {output_tensor.shape}")
    print(f"  Data Type: {output_tensor.dtype}")
    # You can print a snippet of the data or further process it
    print(f"  First 5 elements: {output_tensor.flatten()[:500]}")
    print("-" * 20)
# Example of a simplified post-processing and visualization (may vary based on YOLO version)
# This assumes a typical YOLO output format with bounding boxes, scores, and class IDs
# You might need a more sophisticated NMS and decoding process depending on your model.

# Assuming 'outputs' contains a tensor with detections
# For demonstration, let's assume 'outputs[0]' contains [x1, y1, x2, y2, confidence, class_id]
detections = outputs[0][0] # Assuming batch size 1

# Load class names (e.g., from a coco-classes.txt file)
with open('best-labels.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

for det in detections:
    print(det[0],det[1],det[2])
    x1, y1, x2, y2, conf, class_id = det[:6] # Extract relevant info
    if conf > 0.5: # Confidence threshold
        # Scale coordinates back to original image size
        x1_orig = int(x1 * original_w / input_size)
        y1_orig = int(y1 * original_h / input_size)
        x2_orig = int(x2 * original_w / input_size)
        y2_orig = int(y2 * original_h / input_size)

        print(class_id)
        label = f"{classes[int(class_id)]}: {conf:.2f}"
        cv2.rectangle(image, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
        cv2.putText(image, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow("YOLO ONNX Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()