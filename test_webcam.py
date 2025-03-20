import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
else:
    print("Webcam opened successfully")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Webcam", frame)
        cv2.waitKey(0)  # Wait until a key is pressed
    else:
        print("Failed to read frame")
cap.release()
cv2.destroyAllWindows()
