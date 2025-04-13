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

client = genai.Client(api_key="API KEY")

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