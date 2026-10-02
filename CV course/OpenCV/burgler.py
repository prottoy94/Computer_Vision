import cv2
import time
from pathlib import Path

cap = cv2.VideoCapture('Theif_video.mp4')
output_dir = Path(__file__).parent / 'motion_images'
output_dir.mkdir(exist_ok=True)

frames=[]
gap=5
count=0
last_saved_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(grayscale)
    count += 1

    if len(frames) > gap+1:
        frames.pop(0)
    
    cv2.putText(frame, f'Frame Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    if len(frames) > gap:
        diff = cv2.absdiff(frames[-1], frames[0])
        _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY) # Create a binary image to highlight the differences
        contours,_= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            if cv2.contourArea(c) > 1000:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Motion Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        motion= any(cv2.contourArea(c) > 1000 for c in contours)
        
        if motion:
            cv2.putText(frame, 'Motion Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
            current_time = time.monotonic()
            if current_time - last_saved_time >= 1.0:
                image_path = output_dir / f'motion_frame_{count}.jpg'
                if not cv2.imwrite(str(image_path), frame):
                    print(f'Could not save motion frame to {image_path}')
                last_saved_time = current_time
    
    cv2.imshow('Motion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()