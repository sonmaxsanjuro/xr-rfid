#run as "python run_yolo_inference_image.py <onnx model with full path> <image with full path>"

from ultralytics import YOLO
import cv2
import sys


print("Trying out yolo")
# Load a pre-trained YOLO model (e.g., YOLOv8n)
model = YOLO(sys.argv[1]) 

image_path = sys.argv[2]
results = model.predict(source=image_path, conf=0.5) # conf is confidence threshold
# Process and display results (e.g., using OpenCV)
for r in results:
    im_array = r.plot()  # plot results on the image
    cv2.imshow("YOLO Detection", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
