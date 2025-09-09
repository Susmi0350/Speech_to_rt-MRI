import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random, math
from transformers import Wav2Vec2Processor
import librosa    

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader (lazy loading)."""
    def __init__(self, data, subjects_dict, data_type="train", args=None):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.args = args
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        # Load templates once (small size)
        #template_file = os.path.join(args.data_folder_path, args.dataset, args.template_file)
        template_file = os.path.normpath(os.path.join(args.data_folder_path, args.dataset, args.template_file))
        #print("Template file path:", template_file)

        with open(template_file, 'rb') as fin:
            self.templates = pickle.load(fin, encoding='latin1')

    def __getitem__(self, index):
        """Returns one data pair (lazy loaded)."""
        sample = self.data[index]
        file_name = sample["name"]
        subject_id = sample["subject_id"]

        # Load audio
        wav_path = sample["wav_path"]
        speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values)

        # Load vertice (MRI)
        vertice_path = sample["vertice_path"]
        mri_array = np.load(vertice_path, allow_pickle=True).astype(np.float32) / 255.0
        # max_len = 5000
        # if mri_array.shape[0] < max_len:
        #     pad_len = max_len - mri_array.shape[0]
        #     mri_array = np.vstack([mri_array, np.zeros((pad_len, mri_array.shape[1]), dtype=np.float32)])
        # elif mri_array.shape[0] > max_len:
        #     mri_array = mri_array[:max_len]

        # Load template
        template = self.templates[subject_id].reshape((-1))

        # One-hot
        if self.data_type == "train":
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject_id)]
        else:
            one_hot = self.one_hot_labels  # All zeros for val/test

        return torch.FloatTensor(input_values), torch.FloatTensor(mri_array), \
               torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len


def read_data(args):
    print("Indexing data paths...")
    data = []
    audio_path = os.path.join(args.data_folder_path, args.dataset, args.wav_path)
    vertices_path = os.path.join(args.data_folder_path, args.dataset, args.vertices_path)

    subjects_dict = {
        "train": args.train_subjects.split(" "),
        "val": args.val_subjects.split(" "),
        "test": args.test_subjects.split(" ")
    }

    splits = {
        'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
        'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)},
        'rt-mri': {'train': range(1, 17), 'val': range(17, 20), 'test': range(1, 17)}
    }
    
    # splits = {
    #     'vocaset': {'train': range(1, 41), 'val': range(21, 41), 'test': range(21, 41)},
    #     'BIWI': {'train': range(1, 33), 'val': range(33, 37), 'test': range(37, 41)},
    #     'rt-mri': {'train': range(1, 17), 'val': range(17, 20), 'test': range(20, 22)}
    # }

    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if not f.endswith("wav"):
                continue
            key = f.replace(".wav", ".npy")
            subject_id = f.split("_")[0]
            sentence_id = int(f.split("_")[-2])

            # Verify if vertice file exists
            vertice_path = os.path.join(vertices_path, key)
            if not os.path.exists(vertice_path):
                continue

            entry = {
                "name": f,
                "wav_path": os.path.join(r, f),
                "vertice_path": vertice_path,
                "subject_id": subject_id,
                "sentence_id": sentence_id
            }

            if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
                entry["split"] = "train"
                data.append(entry)
            elif subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
                entry["split"] = "val"
                data.append(entry)
            elif subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
                entry["split"] = "test"
                data.append(entry)

    # Split data
    train_data = [d for d in data if d["split"] == "train"]
    val_data = [d for d in data if d["split"] == "val"]
    test_data = [d for d in data if d["split"] == "test"]

    print(len(train_data), len(val_data), len(test_data))
    return train_data, val_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    dataset["train"] = data.DataLoader(
        dataset=Dataset(train_data, subjects_dict, "train", args), batch_size=1, shuffle=True)
    dataset["valid"] = data.DataLoader(
        dataset=Dataset(valid_data, subjects_dict, "val", args), batch_size=1, shuffle=False)
    dataset["test"] = data.DataLoader(
        dataset=Dataset(test_data, subjects_dict, "test", args), batch_size=1, shuffle=False)

    return dataset
