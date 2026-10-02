import cv2
from ultralytics import YOLO

model = YOLO('yolo26n.pt')  # Load a pretrained YOLO26n model

image=cv2.imread('test_image.jpg')  # Read the input image
results = model.predict(image)  # Perform object detection on the image
annotated_image = results[0].plot()  # Get the annotated image with detected objects

cv2.imshow('YOLO Object Detection', annotated_image)  # Display the image with detected objects
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close all OpenCV windows