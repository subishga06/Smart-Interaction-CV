import streamlit as st
import cv2
from ultralytics import YOLO

# Set up the title and description of your app
st.title("Real-Time Object Detection")
st.write("This application uses YOLOv8 to perform real-time object detection.")

# Load the YOLO model (make sure 'yolov8n.pt' is in your project folder or provide a full path)
model = YOLO("yolov8n.pt")

# Open the webcam (device index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open webcam")
else:
    st.success("Webcam successfully opened")

# Create a placeholder for the video feed
frame_placeholder = st.empty()

# Use Streamlit session state to track if the stop button was pressed
if "stop" not in st.session_state:
    st.session_state.stop = False

# Create a stop button (placed outside the main loop)
if st.button("Stop", key="stop_button"):
    st.session_state.stop = True

# Optional: add a slider for adjusting confidence threshold (this is just for UI demonstration)
conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

# Main loop: process webcam frames until the stop condition is met
while cap.isOpened() and not st.session_state.stop:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab a frame")
        break

    # Run detection on the frame using YOLO
    results = model(frame)
    for result in results:
        # Draw detection overlays (bounding boxes, labels, etc.)
        frame = result.plot()

    # Convert the frame from BGR (OpenCV format) to RGB for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update the placeholder with the new frame
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

# Cleanup: release the webcam and display a final message
cap.release()
st.write("Webcam released, application stopped.")
