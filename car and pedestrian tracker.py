import cv2

# Our Image
img_file = 'Car Image.jpg'
# video = cv2.VideoCapture('MotoVlog_360.mp4')
video = cv2.VideoCapture('video360p.mp4')
# video = cv2.VideoCapture('crash_video360p.mp4')
# video = cv2.VideoCapture('Tesla Dashcam Accident.mp4')

# Our pre-trained car classifier
car_tracker_file = 'car_detector.xml'
# Our pre-trained pedestrian classifier
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
# create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


# Running the while loop forever
while True:

    # Read the current frame
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break    

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around the pedestrains
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 255), 2)
    

    # Display the image with the car spotted
    cv2.imshow('Car and Pedestrian Detector', frame)

    # Dont autoclose the image wait for a keypress
    key = cv2.waitKey(1)  # Stay on the frame for 1ms

    # Stop if Q key is pressed
    if key==81 or key==113:
        break


# Release the VideoCapture
video.release()

"""
# create opencv image
img = cv2.imread(img_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y),(x+w, y+h), (0, 0, 255), 2)

# Display the image with the car spotted
cv2.imshow('Car Detector', img)

# Dont autoclose the image wait for a keypress
cv2.waitKey()
"""

print("Code Completed!")
