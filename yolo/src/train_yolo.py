#run as "python train_yolo.py <yolo pretrained .pt model with full path> <yolo yaml file with full path>"

from ultralytics import YOLO
import sys

model = YOLO(sys.argv[1]) # Start from a pre-trained model
results = model.train(data=sys.argv[2], epochs=100, imgsz=640)
