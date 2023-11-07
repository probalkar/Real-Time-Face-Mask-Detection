import streamlit as st
from streamlit_webrtc import webrtc_streamer
from aiortc import RTCPeerConnection, RTCIceServer
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from deepface import DeepFace

# Load the trained mask detection model
model = load_model("mask_detection_model.h5")

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a function for real-time mask detection
def detect_mask(frame):
    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))
    resized_frame = img_to_array(resized_frame)
    resized_frame = preprocess_input(resized_frame)
    resized_frame = np.expand_dims(resized_frame, axis=0)

    # Perform prediction
    predictions = model.predict(resized_frame)
    return predictions

# Streamlit web app
st.title("Real-time Mask Detection")

# st.title("My first Streamlit app")
# st.write("Hello, world")


def callback(frame):
    # img = frame.to_ndarray(format="bgr24")
    # img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    # flipped = img[:,::-1,:]

    # Perform mask detection
    predictions = detect_mask(frame)
    label = "Mask" if np.argmax(predictions) == 1 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
    # Display the frame with the label
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]

        # Perform mask detection
        # ... Your mask detection code here ...
        predictions = detect_mask(frame)
        label = "Mask" if np.argmax(predictions) == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # If no mask is detected, estimate age and gender
        if label == "No Mask":
            results = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['age', 'gender'], enforce_detection=False)
            results = results[0]
            age = results['age']
            gender = results['dominant_gender']
            # print(results)
            # print(type(results[0]))

            # Display age and gender estimation
            cv2.putText(frame, f'Age: {age:.1f} years', (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display mask detection result
            # ... Your mask detection display code here ...
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame using Streamlit
        # stframe.image(frame_rgb, channels="RGB", use_column_width=True)
        img = frame_rgb.to_ndarray(format="bgr24")
        flipped = img[:,::-1,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")

ice_configuration = {
    "iceServers": [{"urls": "turn:openrelay.metered.ca:80",
                   "username": "openrelayproject",
                   "credential": "openrelayproject"}]
}

peer_connection = RTCPeerConnection(configuration=ice_configuration)

webrtc_streamer(key="example", video_frame_callback=callback, media_stream_constraints={"video":True, "audio":False}, rtc_configuration=peer_connection)
