from ultralytics import YOLO
#Loading pre-trained YOLO model for hand detection.
#Data from https://universe.roboflow.com/handdetect-p7kf9/hand-detection-1d18n
model = YOLO("yolo26n.pt")
results = model.train(data="Hand Detection.v4i.yolov8/data.yaml",epochs=80,imgsz=640)
