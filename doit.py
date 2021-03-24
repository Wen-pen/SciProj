import cv2
import math
import pyautogui

cascPath = "hand.xml"
casc_path_2 = "palm.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
palmCascade = cv2.CascadeClassifier(casc_path_2)

video_capture = cv2.VideoCapture(0)

video_capture.set(3, 1280)
video_capture.set(4, 720)

while True:
    # Capture frame-by-frame
    
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        pyautogui.moveTo(math.ceil(x+w/2) , math.ceil(y+h/2), duration=0.2)

    palms = palmCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in palms:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()