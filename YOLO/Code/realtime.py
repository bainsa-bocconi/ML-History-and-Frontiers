"""
Real-time Object Detection using YOLO and OpenCV

This script implements real-time object detection using the YOLO (You Only Look Once)
model through a webcam feed. It processes each frame from the webcam, detects objects,
and displays the results with bounding boxes and labels in a window.

Authors:
    - Picciano Alisia
    - Patrikov Martin 
    - Micaletto Giorgio

Requirements:
    - OpenCV (cv2)
    - Ultralytics YOLO
    - A working webcam

The script will:
    1. Initialize YOLO model and webcam capture
    2. Process each frame in real-time
    3. Draw bounding boxes and labels around detected objects
    4. Display confidence scores
    5. Continue until 'q' is pressed

Returns:
    None. Displays a real-time video feed with object detection overlay.
"""

import cv2
from ultralytics import YOLO

model = YOLO('yolov5su.pt')  
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame. Check webcam connection or permissions.")
        break

    results = model(frame)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('YOLO Webcam', frame)

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
