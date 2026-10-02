import cv2
from ultralytics import YOLO

capture = cv2.VideoCapture(0)  # Open the default camera (usually the webcam)
model = YOLO('yolo26n.pt')  # Load a pretrained YOLO26

while True:
    ret, frame= capture.read()  # Read a frame from the camera
    results = model.predict(frame)  # Perform object detection on the frame
    annotated_frame = results[0].plot()  # Get the annotated frame with detected objects
    cv2.imshow('YOLO Object Detection', annotated_frame)  # Display the frame with detected objects
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

capture.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows