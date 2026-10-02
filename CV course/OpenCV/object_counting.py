import cv2
import numpy as np
from ultralytics import YOLO

capture = cv2.VideoCapture('Crowded.mp4')  # Open the default camera (usually the webcam)
model = YOLO('yolo26n.pt')  # Load a pretrained YOLO26
uid=set()  # Initialize a set to store unique IDs of detected objects

cv2.namedWindow('YOLO Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Object Detection', 700, 1000)

while True:
    ret, frame= capture.read()  # Read a frame from the camera
    results = model.track(frame, classes=[0], persist=True, verbose=False)  # Perform object detection on the frame
    annotated_frame = results[0].plot()
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box in results[0].boxes:
            uid.add(int(box.id.item()))

        current_people = len(results[0].boxes)

        cv2.putText(
            annotated_frame,
            f'People Now: {current_people}',
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f'Total Seen: {len(uid)}',
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            2
        )

    cv2.imshow('YOLO Object Detection', annotated_frame)  # Display the frame with detected objects

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break