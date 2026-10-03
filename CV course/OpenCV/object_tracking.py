import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

capture = cv2.VideoCapture('Crowded.mp4')  # Open the default camera (usually the webcam)
model = YOLO('yolo26n.pt')  # Load a pretrained YOLO26
cv2.namedWindow('YOLO Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Object Detection', 700, 1000)

id_map = {}  # Initialize a dictionary to map object IDs to their last seen positions
next_id = 0  # Initialize a variable to assign new IDs to detected objects
trail=defaultdict(lambda: deque(maxlen=30))  # Initialize a dictionary to store the trails of each object
appear=defaultdict(int)  # Initialize a dictionary to store the appearance count of each object

while True:
    ret, frame= capture.read()  # Read a frame from the camera
    if not ret:
        break

    results = model.track(frame, classes=[0], persist=True, verbose=False)  # Perform object detection on the frame
    annorated_frame = frame.copy()  # Create a copy of the frame to draw annotations on
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the bounding boxes of detected objects
        ids = results[0].boxes.id.cpu().numpy()  # Get the IDs of detected objects
        
        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box.astype(int)  # Get the coordinates of the bounding box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))  # Calculate the center of the bounding box
            
            obj_id = int(obj_id)
            appear[obj_id] += 1  # Increment the appearance count for the object
            if appear[obj_id] > 5 and obj_id not in id_map:  # Only draw the trail if the object has appeared for more than 5 frames
                id_map[obj_id] = next_id  # Assign a new ID to the object
                next_id += 1  # Increment the next ID counter
            if obj_id in id_map:
                sid=id_map[obj_id]  # Get the assigned ID for the object
                trail[obj_id].append(center)  # Append the center position to the trail of the object
                cv2.rectangle(annorated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
                cv2.putText(annorated_frame, f'ID: {sid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 2)  # Draw the ID of the object   
                cv2.circle(annorated_frame, center, 5, (0, 0, 255), -1)  # Draw a circle at the center of the bounding box
                points = np.array(trail[obj_id], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(annorated_frame, [points], False, (255, 0, 0), 2)
            
    cv2.imshow('YOLO Object Detection', annorated_frame)  # Display the frame with detected objects and trails
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

capture.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows

                                