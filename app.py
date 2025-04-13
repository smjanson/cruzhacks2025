import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import speech_recognition as sr

audio_value = st.audio_input("")
if audio_value:
    # st.audio(audio_value)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_value.getvalue())
        tmp_path = tmp.name

    # Load the recognizer
    recognizer = sr.Recognizer()

    # # Load the audio file
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)  # Read the entire file
        try:
            text = recognizer.recognize_whisper(audio_data)
            st.success("Transcription:")
            st.write(text)
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")

# st.set_page_config(page_title="YOLOv8 Segmentation", layout="wide")

model = YOLO("yolov8n-seg.pt")


def process_frame(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model(img)[0]
    annotated_frame = img.copy()

    if results.masks is not None:
        masks = results.masks.data
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            color = [254, 0, 0]
            colored_mask = np.stack([mask * c for c in color], axis=-1)
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)

        masks = results.masks.data.cpu().numpy()
        combined_mask = np.max(masks, axis=0).astype(np.uint8)
        ground_mask = 1 - combined_mask
        h, w = ground_mask.shape
        path_points = []

        for y in range(h - 1, 0, -20):
            row = ground_mask[y]
            walkable_indices = np.where(row > 0)[0]
            if walkable_indices.size > 0:
                center_x = int(np.mean(walkable_indices))
                path_points.append((center_x, y))

        for i in range(1, len(path_points)):
            cv2.arrowedLine(annotated_frame, path_points[i - 1], path_points[i], (0, 255, 255), 3, tipLength=0.5)

        ground_color = np.array([50, 200, 50], dtype=np.uint8)
        ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="yolo-seg",
    video_frame_callback=process_frame,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)