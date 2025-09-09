import zipfile
import os

npy_dir = "vertices_npy"       # Folder containing .npy files
npy_zip_path = "all_subjects_npy.zip"  # Name of the zip file to create

# Check if folder exists
if not os.path.exists(npy_dir):
    print(f"Folder '{npy_dir}' does not exist. Please check the path.")
else:
    with zipfile.ZipFile(npy_zip_path, 'w') as zipf:
        for root, _, files in os.walk(npy_dir):
            for file in files:
                if file.endswith(".npy"):
                    filepath = os.path.join(root, file)
                    zipf.write(filepath, arcname=file)
    print(f"✅ Created {npy_zip_path}")


import zipfile
import os

wav_dir = "wav"
wav_zip_path = "all_subjects_wav.zip"

if not os.path.exists(wav_dir):
    print(f"Folder '{wav_dir}' does not exist. Please check the path.")
else:
    with zipfile.ZipFile(wav_zip_path, 'w') as zipf:
        for root, _, files in os.walk(wav_dir):
            for file in files:
                if file.endswith(".wav"):
                    filepath = os.path.join(root, file)
                    zipf.write(filepath, arcname=file)
    print(f"✅ Created {wav_zip_path}")
