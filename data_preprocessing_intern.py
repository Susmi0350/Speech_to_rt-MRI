import os
import cv2
import numpy as np
import subprocess

# Define helper functions
def parse_filename(filename):
    name_parts = filename.replace(".mp4", "").split("_")
    subject = name_parts[0]
    sentence_id = name_parts[2]

    if "r1" in name_parts:
        rep = "A"
    elif "r2" in name_parts:
        rep = "B"
    else:
        rep = "A"

    return f"{subject}_{sentence_id}_{rep}"

def extract_audio_ffmpeg(video_path, audio_output_path):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_output_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def generate_from_video(video_path, output_npy_dir, output_wav_dir):
    os.makedirs(output_npy_dir, exist_ok=True)
    os.makedirs(output_wav_dir, exist_ok=True)

    for file in os.listdir(video_path):
        if not file.endswith(".mp4"):
            continue

        full_path = os.path.join(video_path, file)
        name_base = parse_filename(file)

        # Extract grayscale frames and flatten
        cap = cv2.VideoCapture(full_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten()
            frames.append(flat)
        cap.release()

        frames_np = np.array(frames)
        np.save(os.path.join(output_npy_dir, name_base + ".npy"), frames_np)

        # Extract audio
        audio_output_path = os.path.join(output_wav_dir, name_base + ".wav")
        extract_audio_ffmpeg(full_path, audio_output_path)

# Set base paths
base_path = "/home/ubuntu/projects/Susmita/audio-to-mri-data"
vertices_npy_path = "/home/ubuntu/projects/Susmita/vertices_npy"
wav_path = "/home/ubuntu/projects/Susmita/wav"

# Loop through all 75 subjects
for i in range(1, 76):  # sub001 to sub075
    subject = f"sub{i:03d}"
    subject_video_path = os.path.join(base_path, subject, "2drt", "video")

    if not os.path.exists(subject_video_path):
        print(f"⚠️ Skipping {subject}: Path not found → {subject_video_path}")
        continue

    print(f"✅ Processing: {subject}")
    generate_from_video(subject_video_path, vertices_npy_path, wav_path)

# Summary
print("✅ Done!")
print("Total .npy files:", len(os.listdir(vertices_npy_path)))
print("Total .wav files:", len(os.listdir(wav_path)))
