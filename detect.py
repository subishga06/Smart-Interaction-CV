import cv2
from ultralytics import YOLO

def trigger_notification():
    print("Alert: A person has been detected!")
    # Add any extra actions here (e.g., play a sound, send an email)

print("Starting detection script...")

# Load the YOLO model (adjust the model path if needed)
model = YOLO("yolov8n.pt")

# Open the webcam (device 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam")
else:
    print("Webcam successfully opened")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame")
            break

        # Run detection on the frame
        results = model(frame)
        for result in results:
            # Draw bounding boxes on the frame
            frame = result.plot()

            # Check if any detected box belongs to a "person"
            # The detected boxes are stored in result.boxes (if available)
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # box.cls is a tensor containing the detected class id(s)
                    cls_id = int(box.cls[0])
                    # Access the class name using model.names dictionary
                    if model.names.get(cls_id, "") == "person":
                        trigger_notification()

        # Display the frame in a window
        cv2.imshow("YOLOv8 Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting detection loop...")
            break

except KeyboardInterrupt:
    print("Detection interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
