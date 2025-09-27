#run as "python conv_yolo_2_onnx.py <yolo .pt file with full path>"
from ultralytics import YOLO
import sys

# Load your YOLO .pt model
model = YOLO(sys.argv[1])

# Export the model to ONNX format
model.export(format='onnx', imgsz=640, dynamic=False, simplify=True)