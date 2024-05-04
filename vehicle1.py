import cv2
import numpy as np

# Video capture
cap = cv2.VideoCapture("video.mp4")

min_width_rect = 80  # min width of rectangle
min_height_rect = 80  # min height of rectangle

count_line_position = 550

algo = cv2.createBackgroundSubtractorMOG2()

# Replace this with your actual ground truth bounding boxes
ground_truth_boxes = [(100, 200, 300, 400), (150, 250, 350, 450), ...]

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []

offset = 6  # allowable error between pixel

counter = 0
total_frames = 0
correctly_detected_frames = 0

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Error reading frame.")
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

    for box in ground_truth_boxes:
        for (x, y) in detect:
            if box[1] < y < box[3] and box[0] < x < box[2]:
                iou = calculate_iou((x, y, x + w, y + h), box)
                if iou > 0.5:  # Adjust the IoU threshold as needed
                    correctly_detected_frames += 1
                    break

    total_frames += 1

    cv2.putText(frame1, "Vehicle Counter: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
        break

accuracy = correctly_detected_frames / total_frames
print("Accuracy:", accuracy)

cv2.destroyAllWindows()
cap.release()
