import cv2
from ultralytics import YOLO

# Replace this with the video address shown by your phone camera app.
# Example: http://192.168.0.105:8080/video
phone_camera_url = 'http://192.168.68.104:8080/video'

capture = cv2.VideoCapture(phone_camera_url)  # Open the phone camera
model = YOLO('yolo26n.pt')  # Load a pretrained YOLO26 model

while True:
    ret, frame = capture.read()  # Read a frame from the phone camera

    if not ret:
        print('Could not receive video from the phone.')
        print('Check the phone IP address, URL, and Wi-Fi connection.')
        break

    results = model.predict(frame)  # Perform object detection on the frame
    annotated_frame = results[0].plot()  # Get the annotated frame

    cv2.imshow('Phone Camera Object Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit
        break

capture.release()  # Release the phone camera
cv2.destroyAllWindows()  # Close all OpenCV windows
