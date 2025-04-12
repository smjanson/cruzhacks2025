import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt") 

cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
# count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    results = model(frame)[0]
    annotated_frame = frame.copy()
    if results.masks is not None:
        masks = results.masks.data  # (N, H, W) binary masks
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)  # convert to numpy
            color = [254,0,0]#np.random.randint(0, 255, (3,), dtype=np.uint8)
            colored_mask = np.stack([mask * c for c in color], axis=-1)
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)
        masks = results.masks.data.cpu().numpy()  # shape: (N, H, W), all binary (0 or 1)
        combined_mask = np.max(masks, axis=0).astype(np.uint8)  # shape (H, W)
        ground_mask = 1 - combined_mask
        h, w = ground_mask.shape
        path_points = []

        # Sample Y positions from bottom of image up (like steps)
        for y in range(h - 1, 0, -20):
            row = ground_mask[y]
            walkable_indices = np.where(row > 0)[0]
            if walkable_indices.size > 0:
                center_x = int(np.mean(walkable_indices))
                path_points.append((center_x, y))
        for i in range(1, len(path_points)):
            cv2.arrowedLine(annotated_frame, path_points[i - 1], path_points[i], (0, 255, 255), 3, tipLength=0.5)

        ground_color = np.array([50, 200, 50], dtype=np.uint8)  # light green
        ground_overlay = np.stack([ground_mask * c for c in ground_color], axis=-1)
        annotated_frame = cv2.addWeighted(annotated_frame, 1.0, ground_overlay, 0.3, 0)
        # print(results.masks)
        # for result in results:
        #     xy = result.masks.xy  # mask in polygon format

        #     masks = result.masks.data  # mask in matrix format (num_objects x H x W)
        #     print("MASKS", masks)

    # annotated_frame = results.plot(labels=False, boxes=False)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_placeholder.image(annotated_frame, channels="BGR")
    # count += 1
cap.release()
cv2.destroyAllWindows()
