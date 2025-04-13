import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from gtts import gTTS
from io import BytesIO
from PIL import Image
from google import genai
# import google.generativeai as genai
import os
import json
import tempfile
import speech_recognition as sr

# Page configuration
st.set_page_config(page_title="Navigation Assistant", layout="wide")
st.title("Smart Navigation Guidance")

# Initialize Gemini client
try:
    client = genai.Client(api_key="INSERT GEMINI API KEY")
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    client = None

# Initialize session state variables
if 'audio_feedback' not in st.session_state:
    st.session_state.audio_feedback = True
if "last_instruction_time" not in st.session_state:
    st.session_state.last_instruction_time = 0
if "latest_instruction" not in st.session_state:
    st.session_state.latest_instruction = "Waiting for instruction..."
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "scene_context" not in st.session_state:
    st.session_state.scene_context = {
        "previous_instructions": [],
        "environment_description": "",
        "detected_objects": []
    }
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
            prompt = f"Can you tell me what location the user wants to go to from their transcribe: {text}"
            res = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt]
            )
        except sr.UnknownValueError:
            st.error("Could not understand audio")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
# Load YOLO model
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("yolov8n-seg.pt")
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

model = load_yolo_model()

# Text-to-Speech function
def text_to_speech(text, language="en", slow=False):
    if not text or not st.session_state.audio_feedback:
        return None
    try:
        tts = gTTS(text=text, lang=language, slow=slow)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# Function to extract scene information from YOLO results
def extract_scene_info(results, frame_shape):
    scene_info = {
        "detected_objects": [],
        "obstacles": [],
        "walkable_areas": {},
        "frame_width": frame_shape[1],
        "frame_height": frame_shape[0]
    }
    
    # Extract object information if available
    if hasattr(results, 'boxes') and results.boxes is not None:
        for i, box in enumerate(results.boxes):
            if hasattr(results, 'names') and results.names is not None:
                class_id = int(box.cls.item())
                class_name = results.names[class_id]
                
                # Get confidence score
                confidence = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
                
                # Get box coordinates (normalized format)
                if hasattr(box, 'xyxy'):
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    
                    # Calculate position in the frame (relative to center)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Determine horizontal position
                    relative_x = center_x / frame_shape[1]
                    horizontal_pos = "left" if relative_x < 0.33 else "center" if relative_x < 0.66 else "right"
                    
                    # Determine vertical position (lower y value is higher in the frame)
                    relative_y = center_y / frame_shape[0]
                    vertical_pos = "top" if relative_y < 0.33 else "middle" if relative_y < 0.66 else "bottom"
                    
                    # Object size estimation
                    area = (x2 - x1) * (y2 - y1)
                    area_percentage = area / (frame_shape[0] * frame_shape[1])
                    size = "small" if area_percentage < 0.1 else "medium" if area_percentage < 0.3 else "large"
                    
                    # Add to detected objects
                    obj_info = {
                        "class": class_name,
                        "confidence": confidence,
                        "position": {
                            "horizontal": horizontal_pos,
                            "vertical": vertical_pos
                        },
                        "size": size
                    }
                    
                    scene_info["detected_objects"].append(obj_info)
                    
                    # Treat most objects as obstacles
                    scene_info["obstacles"].append(obj_info)
    
    # Extract walkable area information if masks are available
    if hasattr(results, 'masks') and results.masks is not None:
        h, w = frame_shape[:2]
        
        # Analyze walkable areas by dividing the image into a grid
        grid_size = 3  # 3x3 grid
        for x_section in range(grid_size):
            for y_section in range(grid_size):
                # Define section boundaries
                x_start = int(x_section * w / grid_size)
                x_end = int((x_section + 1) * w / grid_size)
                y_start = int(y_section * h / grid_size)
                y_end = int((y_section + 1) * h / grid_size)
                
                # Map these sections to named areas
                section_names = {
                    (0, 0): "top_left", (1, 0): "top_center", (2, 0): "top_right",
                    (0, 1): "middle_left", (1, 1): "middle_center", (2, 1): "middle_right",
                    (0, 2): "bottom_left", (1, 2): "bottom_center", (2, 2): "bottom_right"
                }
                
                section_name = section_names.get((x_section, y_section), f"section_{x_section}_{y_section}")
                
                # Determine if this section is walkable (has no masks)
                section_walkable = True
                
                # Check if there are objects in this section
                obstacles_in_section = []
                for obj in scene_info["obstacles"]:
                    if (obj["position"]["horizontal"] == "left" and x_section == 0) or \
                       (obj["position"]["horizontal"] == "center" and x_section == 1) or \
                       (obj["position"]["horizontal"] == "right" and x_section == 2):
                        if (obj["position"]["vertical"] == "top" and y_section == 0) or \
                           (obj["position"]["vertical"] == "middle" and y_section == 1) or \
                           (obj["position"]["vertical"] == "bottom" and y_section == 2):
                            section_walkable = False
                            obstacles_in_section.append(obj["class"])
                
                scene_info["walkable_areas"][section_name] = {
                    "walkable": section_walkable,
                    "obstacles": obstacles_in_section
                }
    
    return scene_info

# Function to get navigation instructions from Gemini with enhanced scene context
def get_gemini_instructions(frame, scene_info=None, context=None):
    if client is None:
        return "Gemini API not available"
    
    try:
        # Save current frame to temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            img_path = tmp_file.name
            cv2.imwrite(img_path, frame)
            
        # Convert to format expected by Gemini
        gemini_image = Image.open(img_path).convert("RGB")
        
        # Build prompt with scene information
        prompt = f"""You are an AI navigation assistant helping a visually impaired person navigate through their environment. 
        
Your task is to provide clear, concise navigation instructions based on what you see in the camera feed.

Analyze the current scene and provide ONLY a brief instruction (5-15 words maximum) about where the person should go next to avoid obstacles and navigate safely.
They should also end up somewhere around their destination which is at {res}.
DO NOT describe the scene or what you see - ONLY give a clear navigation instruction.

Examples of good responses:
- "Move slightly right to avoid obstacle ahead"
- "Continue straight, clear path"
- "Stop and wait, obstacle directly ahead"
- "Turn left, doorway available"
- "Step carefully over small object ahead"

Your instruction should be clear, actionable, and focused only on the next immediate step.
"""
        
        # Add scene information to the prompt if available
        if scene_info:
            scene_json = json.dumps(scene_info, indent=2)
            prompt += f"\nHere is technical information about the scene to help you make better decisions:\n```\n{scene_json}\n```\n"
            
        # Add historical context if available
        if context and 'previous_instructions' in context and context['previous_instructions']:
            prompt += "\nYour previous instructions were:\n"
            for prev_instr in context['previous_instructions'][-3:]:  # Last 3 instructions
                prompt += f"- {prev_instr}\n"
        
        # Send to Gemini for analysis
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, gemini_image]
        )
        
        instruction = response.text.strip()
        
        # Clean up temp file
        os.unlink(img_path)
        
        # Update context
        if context and 'previous_instructions' in context:
            context['previous_instructions'].append(instruction)
            # Keep only last 5 instructions
            if len(context['previous_instructions']) > 5:
                context['previous_instructions'] = context['previous_instructions'][-5:]
        
        return instruction
        
    except Exception as e:
        st.sidebar.error(f"Gemini error: {e}")
        return "Unable to analyze environment"

# Start/Stop camera button
start_button = st.button(
    "Start Camera" if not st.session_state.run_camera else "Stop Camera",
    type="primary" if not st.session_state.run_camera else "secondary"
)

if start_button:
    st.session_state.run_camera = not st.session_state.run_camera
    # Reset context when starting
    if st.session_state.run_camera:
        st.session_state.scene_context = {
            "previous_instructions": [],
            "environment_description": "",
            "detected_objects": []
        }

# Audio feedback toggle
audio_enabled = st.sidebar.checkbox("Enable Audio Feedback", value=True)
st.session_state.audio_feedback = audio_enabled

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    instruction_interval = st.slider("Instruction Interval (seconds)", 1, 10, 10)

# Create placeholders for video and audio
video_placeholder = st.empty()
instruction_placeholder = st.empty()
audio_placeholder = st.empty()

# Instructions
st.sidebar.markdown("## How to use")
st.sidebar.markdown("1. Click 'Start Camera' to begin")
st.sidebar.markdown("2. The system will identify obstacles and highlight a safe path")
st.sidebar.markdown("3. Navigation instructions will be provided every few seconds")
st.sidebar.markdown("4. Toggle audio feedback on/off as needed")

# Explanation of visual elements
# st.sidebar.markdown("## Visual Guide")
# st.sidebar.markdown("- 🟥 Red: Obstacles detected")
# st.sidebar.markdown("- 🟩 Green: Safe walkable area")
# st.sidebar.markdown("- 🟨 Yellow Arrows: Suggested path")

if st.session_state.run_camera:
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        st.info("Camera started. Processing video...")
        
        try:
            # Main loop for camera processing
            while st.session_state.run_camera:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                
                # Process with YOLO if model is loaded
                scene_info = None
                if model:
                    # Make a copy for annotation
                    annotated_frame = frame.copy()
                    
                    # Run YOLO detection
                    results = model(frame)[0]
                    path_points = []
                    
                    # Extract scene information for Gemini
                    scene_info = extract_scene_info(results, frame.shape)
                    
                    # Process segmentation masks if available
                    if results.masks is not None:
                        # Get frame dimensions
                        h, w = frame.shape[:2]
                        
                        try:
                            # Process each mask and ensure it has the correct dimensions
                            masks = results.masks.data.cpu().numpy()
                            
                            # Create an empty combined mask with the same dimensions as the frame
                            combined_mask = np.zeros((h, w), dtype=np.uint8)
                            
                            # Iterate through each mask and resize if needed
                            for i, mask_data in enumerate(masks):
                                # Convert to proper format
                                mask = mask_data.astype(np.uint8)
                                
                                # Resize mask if dimensions don't match the frame
                                mask_h, mask_w = mask.shape[:2]
                                if mask_h != h or mask_w != w:
                                    mask = cv2.resize(mask, (w, h))
                                
                                # Add to combined mask (union of all masks)
                                combined_mask = np.maximum(combined_mask, mask)
                                
                                # Create colored mask for overlay
                                color = [254, 0, 0]  # Red for obstacles
                                colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
                                for c in range(3):
                                    colored_mask[:, :, c] = mask * color[c]
                                
                                # Overlay on frame
                                annotated_frame = cv2.addWeighted(
                                    annotated_frame, 1.0, colored_mask, 0.5, 0
                                )
                            
                            # Create ground mask (inverse of combined mask)
                            ground_mask = 1 - combined_mask
                            
                            # Find a path through the walkable area
                            # Sample every 20 pixels vertically, find center of walkable area horizontally
                            for y in range(h - 20, 0, -20):  # Start from bottom of image
                                if y >= ground_mask.shape[0]:
                                    continue
                                
                                row = ground_mask[y]
                                walkable_indices = np.where(row > 0)[0]
                                
                                if walkable_indices.size > 0:
                                    center_x = int(np.mean(walkable_indices))
                                    path_points.append((center_x, y))
                            
                            # Draw path with arrows
                            for i in range(1, len(path_points)):
                                cv2.arrowedLine(
                                    annotated_frame, 
                                    path_points[i - 1], 
                                    path_points[i], 
                                    (0, 255, 255),  # Yellow for path
                                    3, 
                                    tipLength=0.5
                                )
                            
                            # Create green overlay for walkable areas
                            ground_color = np.array([50, 200, 50], dtype=np.uint8)  # Green
                            ground_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                            for c in range(3):
                                ground_overlay[:, :, c] = ground_mask * ground_color[c]
                            
                            # Apply ground overlay
                            annotated_frame = cv2.addWeighted(
                                annotated_frame, 1.0, ground_overlay, 0.3, 0
                            )
                            
                        except Exception as e:
                            st.sidebar.error(f"Mask processing error: {e}")
                    
                    # Display the processed frame
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                else:
                    # If YOLO model isn't loaded, just show the raw frame
                    video_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Generate navigation instructions at intervals
                current_time = time.time()
                if current_time - st.session_state.last_instruction_time >= instruction_interval:
                    st.session_state.last_instruction_time = current_time
                    
                    # Get instructions from Gemini with scene information and context
                    instruction = get_gemini_instructions(
                        frame, 
                        scene_info=scene_info, 
                        context=st.session_state.scene_context
                    )
                    st.session_state.latest_instruction = instruction
                    
                    # Update the instruction on the UI
                    instruction_placeholder.markdown("### 🗣 Navigation Instruction:")
                    instruction_placeholder.success(instruction)
                    
                    # Provide audio feedback
                    if st.session_state.audio_feedback:
                        audio_file = text_to_speech(instruction)
                        if audio_file:
                            audio_placeholder.audio(audio_file, format="audio/mp3")
                
                # Add a small delay to prevent UI lag
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            
        finally:
            # Always release the camera when done
            cap.release()
            st.session_state.run_camera = False
            st.info("Camera stopped.")
else:
    # Show a placeholder when camera is not running
    video_placeholder.info("Click 'Start Camera' to begin navigation assistance")

# Display historical context
with st.expander("Navigation History"):
    if 'previous_instructions' in st.session_state.scene_context and st.session_state.scene_context['previous_instructions']:
        for i, instr in enumerate(reversed(st.session_state.scene_context['previous_instructions'])):
            st.text(f"{len(st.session_state.scene_context['previous_instructions']) - i}. {instr}")
    else:
        st.info("No navigation instructions yet.")