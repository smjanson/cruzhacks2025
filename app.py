# import streamlit as st
# from streamlit_webrtc import webrtc_streamer
# import av
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import tempfile
# import speech_recognition as sr
# from transformers import AutoModelForCausalLM, AutoProcessor, WhisperProcessor, WhisperForConditionalGeneration
# from gtts import gTTS
# from io import BytesIO


# from PIL import Image
# import torch
# import time
# from google import genai
# from google.genai import types

# client = genai.Client(api_key="AIzaSyB9VY710U-PVkH8IqOnTnM-2UgCkrJoWqM")


# # if 'audio_feedback' not in st.session_state:
# #     st.session_state.audio_feedback = True

# # def text_to_speech(text, language="en", slow=False):
# #     if not text or not st.session_state.audio_feedback:
# #         return None
# #     try:
# #         tts = gTTS(text=text, lang=language, slow=slow)
# #         audio_file = BytesIO()
# #         tts.write_to_fp(audio_file)
# #         audio_file.seek(0)
# #         return audio_file
# #     except Exception as e:
# #         print(f"TTS Error: {e}")
# #         return None

# # def play_audio_feedback(message):
# #     audio_file = text_to_speech(message)
# #     if audio_file:
# #         if 'audio_placeholder' not in st.session_state:
# #             st.session_state.audio_placeholder = st.empty()
# #         st.session_state.audio_placeholder.audio(audio_file, format="audio/mp3")

# # play_audio_feedback("hi im sierra apple banana kiwi lemon lime mango guava juice")

# prompt = "Please limit response to less than 10 words at a time." # find path
# prompt_user_portion = ""

# audio_value = st.audio_input(".")
# if audio_value:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(audio_value.getvalue())
#         tmp_path = tmp.name

#     recognizer = sr.Recognizer()

#     with sr.AudioFile(tmp_path) as source:
#         audio_data = recognizer.record(source)  # Read the entire file
#         try:
#             text = recognizer.recognize_whisper(audio_data)
#             prompt_user_portion = text
#             # st.success("Transcription:")
#             # st.write(text)
#         except sr.UnknownValueError:
#             st.error("Could not understand audio")
#         except sr.RequestError as e:
#             st.error(f"Could not request results; {e}")

# # st.set_page_config(page_title="YOLOv8 Segmentation", layout="wide")

# model = YOLO("yolov8n-seg.pt")
# last_processed_time = 0
# process_interval = 0.5  # seconds
# instruction_placeholder = st.empty()
# instruction_placeholder.markdown("### tesitng")

# def process_frame(frame):
#     img = frame.to_ndarray(format="bgr24")

#     global last_processed_time
#     global process_interval
#     current_time = time.time()
#     if current_time - last_processed_time >= process_interval:
#         last_processed_time = current_time

#     #     # Save the current frame to a temporary file
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
#             img_path = tmp_file.name
#             cv2.imwrite(img_path, frame)
#         print(img_path)
#         try:
#             response = client.models.generate_content(
#                 model="gemini-2.0-flash",
#                 contents=["You are helping a blind person navigate through this environment. " \
#                 "Based on what you see, what direction should they move in to avoid obstacles in a " \
#                 "short description?", Image.open(img_path).convert("RGB")])
#             instruction = response.text
#         except Exception as e:
#             instruction = f"Error from Gemini: {e}"
#         print(instruction)
#     # results = model(img)[0]
#     # annotated_frame = img.copy()

#     # if results.masks is not None:
#     #     #######################
#     #     # OBJECT ANNOTATION ---
#     #     #######################
#     #     masks = results.masks.data
#     #     for mask in masks:
#     #         mask = mask.cpu().numpy().astype(np.uint8)
#     #         color = [254, 0, 0]
#     #         colored_mask = np.stack([mask * c for c in color], axis=-1)
#     #         annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
#     #     masks = results.masks.data.cpu().numpy()
#     #     combined_mask = np.max(masks, axis=0).astype(np.uint8)
#     #     ground_mask = 1 - combined_mask
#     #     h, w = ground_mask.shape
#     #     path_points = []

#     #     ########################
#     #     # ARROW NOTATION ------
#     #     ########################
#     #     for y in range(h - 1, 0, -20):
#     #         row = ground_mask[y]
#     #         walkable_indices = np.where(row > 0)[0]
#     #         if walkable_indices.size > 0:
#     #             center_x = int(np.mean(walkable_indices))
#     #             path_points.append((center_x, y))

#     #     for i in range(1, len(path_points)):
#     #         cv2.arrowedLine(annotated_frame, path_points[i - 1], path_points[i], (0, 255, 255), 3, tipLength=0.5)

#     #     ####################
#     #     # GROUND OVERLAY ---
#     #     ####################
#     #     ground_color = np.array([50, 200, 50], dtype=np.uint8)
#     #     ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
#         # annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)

#     return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# webrtc_streamer(
#     key="yolo-seg",
#     video_frame_callback=process_frame,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )
import streamlit as st
import cv2
import torch
from PIL import Image
import tempfile
import time
from google import genai
from google.genai import types

# UI setup
st.set_page_config(page_title="Navigation Assistant", layout="wide")
st.title("Smart Navigation Guidance")

client = genai.Client(api_key="AIzaSyB9VY710U-PVkH8IqOnTnM-2UgCkrJoWqM")

# Placeholders for video and instruction
video_placeholder = st.empty()
instruction_placeholder = st.empty()

# Set up video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open webcam.")
    st.stop()

# Define the smart prompt
prompt = (
    "You are helping a blind person navigate through this environment. "
    "Based on what you see, describe clearly what direction they should move in to avoid obstacles."
)

# Loop variables
last_processed_time = 0
process_interval = 5  # seconds

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        # Show the live video
        video_placeholder.image(frame, channels="BGR", use_container_width=True)

        current_time = time.time()
        if current_time - last_processed_time >= process_interval:
            last_processed_time = current_time

            # Save the current frame to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                img_path = tmp_file.name
                cv2.imwrite(img_path, frame)
            print(img_path)
            # Load image using PIL
            image = Image.open(img_path).convert("RGB")

            # Send to Gemini
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=["You are helping a blind person navigate through this environment. " \
                    "Based on what you see, what direction should they move in to avoid obstacles in a " \
                    "short description?", image])
                instruction = response.text
            except Exception as e:
                instruction = f"Error from Gemini: {e}"

            # Show navigation instruction
            instruction_placeholder.markdown("### 🗣 Navigation Instruction (Updated every 5s):")
            instruction_placeholder.success(instruction)

        # Small delay to reduce CPU usage (streamlit will still rerun frame)
        time.sleep(0.05)

except KeyboardInterrupt:
    cap.release()
    st.stop()
finally:
    cap.release()