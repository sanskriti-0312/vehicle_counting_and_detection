import cv2
import numpy as np

#web camera

cap = cv2.VideoCapture("video.mp4")

min_width_rect = 80 #min width of rectangle
min_height_rect = 80 #min height of rectangle

count_line_position = 550

#initialize substractor

algo = cv2.createBackgroundSubtractorMOG2()

ground_truth = [2, 3, 4, 5, ...]

def center_handle(x,y,w,h):
    x1 = int (w/2)
    y1 = int (h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []

offset = 6 #allowable error between pixel

counter = 0

while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3,3), 5)
    
    #applying on each frame
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255,127,0), 3)
    
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h),(0,255,0), 2)
        
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0,0,255), -1)
        
        for (x,y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
            cv2.line(frame1, (25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("Vehicle counter : "+str(counter) )
     # Add a ground truth array (replace this with your actual ground truth data)
    ground_truth = [2, 3, 4, 5, ...]

    # ... (rest of the code remains unchanged)

    #  Inside the loop where you update the counter, compare with ground truth
    for (x, y) in detect:
        if y < (count_line_position + offset) and y > (count_line_position - offset):
            counter += 1

    cv2.putText(frame1, "Vehicle Counter : "+str(counter),(450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
    
    #cv2.imshow('detector', dilatada)
    
    #if ret:
    cv2.imshow('Video Original', frame1)
    
    if cv2.waitKey(1) == 13:
        break
    
cv2.destroyAllWindows()
cap.release()


















'''Accuracy Calculation:
accuracy can be calculated by comparing the detected vehicles with a ground truth dataset. 
Intersection over Union (IoU) is used to evaluate the overlap between detected bounding boxes and ground 
truth bounding boxes.'''








'''# Video capture
cap = cv2.VideoCapture("video.mp4")

while True:
    # Read a frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error reading frame.")
        break

    # Display the frame
    cv2.imshow('Video Original', frame)

    # Exit on 'Enter' key press
    if cv2.waitKey(1) == 13:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()'''


'''# Video capture
cap = cv2.VideoCapture("video.mp4")

# Print video properties
print("Frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames per second:", cap.get(cv2.CAP_PROP_FPS))

while True:
    # Read a frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error reading frame.")
        break

    # Display the frame
    cv2.imshow('Video Original', frame)

    # Exit on 'Enter' key press
    if cv2.waitKey(1) == 13:
        break

# Release resources
cv2.destroyAllWindows()
cap.release()'''

