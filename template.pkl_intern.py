import os
import cv2
import numpy as np
import pickle

def flatten_frame(frame):
    """Convert BGR frame to flattened grayscale 1D array."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.flatten()

def get_first_video(subject_dir):
    """Get the first video (sorted order) for a subject directory."""
    if not os.path.exists(subject_dir):
        return None
    videos = [f for f in os.listdir(subject_dir) if f.endswith('.mp4')]
    videos.sort()
    return os.path.join(subject_dir, videos[0]) if videos else None

def create_templates(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output folder if not exist

    for i in range(1, 76):  # sub001 to sub075
        subject_id = f"sub{i:03d}"
        subject_video_path = os.path.join(root_dir, subject_id, "2drt", "video")

        if not os.path.exists(subject_video_path):
            print(f"❌ Path not found for {subject_id}: {subject_video_path}")
            continue

        first_video_path = get_first_video(subject_video_path)
        if not first_video_path:
            print(f"⚠️ No videos found for {subject_id}")
            continue

        cap = cv2.VideoCapture(first_video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"⚠️ Couldn't read frame from {first_video_path}")
            continue

        flattened = flatten_frame(frame)
        template_path = os.path.join(output_dir, f"{subject_id}_template.pkl")

        with open(template_path, 'wb') as f:
            pickle.dump({'subject': subject_id, 'template': flattened}, f)

        print(f"✅ Saved template for {subject_id} at {template_path}")

# === RUN THIS CELL IN JUPYTER NOTEBOOK ===
# Adjust 'root_dir' to the folder where your data is located on your local machine
create_templates(
    root_dir="audio-to-mri-data",  # e.g. relative path in your current directory
    output_dir="templates"         # output folder for templates
)
