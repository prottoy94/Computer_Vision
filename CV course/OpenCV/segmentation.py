import cv2
import numpy as np
from ultralytics import YOLO

capture = cv2.VideoCapture('Crowded.mp4')  # Open the default camera (usually the webcam)
model = YOLO('yolo26n-seg.pt')  # Load a pretrained YOLO26 segmentation model
cv2.namedWindow('YOLO Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Object Detection', 700, 1000)

while True:
    ret, frame= capture.read()  # Read a frame from the camera
    if not ret:
        break
    results = model.track(frame, classes=[0], persist=True, verbose=False)  # Perform object detection on the frame
   
    
    for r in results:
         annorated_frame = frame.copy()  # Create a copy of the frame to draw annotations on
         if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            masks = r.masks.data.cpu().numpy()  # Get the masks of detected objects
            boxes = r.boxes.xyxy.cpu().numpy()  # Get the bounding boxes of detected objects
            ids = r.boxes.id.cpu().numpy()  # Get the IDs of detected objects
            
            for i, mask in enumerate(masks):
                person_id=int(ids[i])
                x1, y1, x2, y2 = boxes[i].astype(int)  # Get the coordinates of the bounding box
                mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0]))  # Resize the mask to match the frame size
                countours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annorated_frame, countours, -1, (0 , 255, 0), 2)  # Draw the contours of the mask
                #cv2.rectangle(annorated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
                cv2.putText(annorated_frame, f'ID: {person_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)  # Draw the ID of the object
    cv2.imshow('YOLO Object Detection', annorated_frame)  # Display the frame with detected objects and trails
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break
capture.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
    
                