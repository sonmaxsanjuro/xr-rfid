#run as "python run_yolo_inference_cam.py <yolo .py model with full path>"
from ultralytics import YOLO
import cv2
import sys


print("Trying out yolo")
# Load a pre-trained YOLO model (e.g., YOLOv8n)
model = YOLO("sys.argv[1]") 

# For webcam:
cap = cv2.VideoCapture(0) # 0 for default webcam

# For a video file:
# cap = cv2.VideoCapture("path/to/your/video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5)
    # Display results
    for r in results:
        im_array = r.plot()
        cv2.imshow("YOLO Detection", im_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
