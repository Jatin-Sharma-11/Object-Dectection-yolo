# detector.py

from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.names = self.model.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            name = self.names[class_id]
            detections.append({
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'confidence': conf,
                'class_id': class_id,
                'name': name
            })
        return detections
