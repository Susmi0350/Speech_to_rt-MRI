# with sliding
import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# def split_into_chunks(audio, vertice, chunk_size=600, stride=300):
#     audio_chunks, vertice_chunks = [], []
#     max_len = min(audio.shape[0], vertice.shape[0])
#     for start in range(0, max_len - chunk_size + 1, stride):
#         audio_chunks.append(audio[start:start+chunk_size])
#         vertice_chunks.append(vertice[start:start+chunk_size])
#     return audio_chunks, vertice_chunks

def split_into_chunks(audio, vertice, chunk_size_mri=600, stride_mri=300, mri_fps=83, audio_sr=16000):
    audio_chunks, vertice_chunks = [], []

    # Compute durations in seconds
    chunk_duration = chunk_size_mri / mri_fps         # ~7.23 sec
    stride_duration = stride_mri / mri_fps            # ~3.61 sec

    # Compute sample counts
    audio_chunk_size = int(chunk_duration * audio_sr)  # ~115680 samples
    audio_stride = int(stride_duration * audio_sr)     # ~57840 samples

    # Get min valid time range based on both modalities
    max_mri_len = vertice.shape[0]
    max_audio_len = audio.shape[0]
    max_chunks = (max_mri_len - chunk_size_mri) // stride_mri + 1

    for i in range(max_chunks):
        start_mri = i * stride_mri
        end_mri = start_mri + chunk_size_mri

        start_audio = i * audio_stride
        end_audio = start_audio + audio_chunk_size

        if end_audio <= max_audio_len and end_mri <= max_mri_len:
            audio_chunks.append(audio[start_audio:end_audio])
            vertice_chunks.append(vertice[start_mri:end_mri])
        else:
            break  # Avoid incomplete chunks

    return audio_chunks, vertice_chunks

def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=1):
    save_path = os.path.join(args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0
    for e in range(epoch+1):
        loss_log = []
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            audio_chunks, vertice_chunks = split_into_chunks(audio.squeeze(0), vertice.squeeze(0), chunk_size_mri=200, stride_mri=100)
            for audio_chunk, vertice_chunk in zip(audio_chunks, vertice_chunks):
                audio_batch = audio_chunk.unsqueeze(0).to(device="cuda")
                vertice_batch = vertice_chunk.unsqueeze(0).to(device="cuda")
                template_batch = template.to(device="cuda")
                one_hot_batch = one_hot.to(device="cuda")

                loss = model(audio_batch, template_batch, vertice_batch, one_hot_batch, criterion, teacher_forcing=False)

                #addd this if co  ndition
                if torch.isnan(loss) or torch.isinf(loss):
                    print("🔴 Detected NaN or Inf in loss.")
                    print("File name:", file_name)
                    print("Audio min/max:", audio_batch.min().item(), audio_batch.max().item())
                    print("MRI target min/max:", vertice_batch.min().item(), vertice_batch.max().item())
                    exit()
            
                loss.backward()
                loss_log.append(loss.item())
                if i % args.gradient_accumulation_steps == 0:
                    optimizer.step() 
                    optimizer.zero_grad()

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.7f}".format((e+1), iteration, np.mean(loss_log)))

        valid_loss_log = []
        model.eval()
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            audio_chunks, vertice_chunks = split_into_chunks(audio.squeeze(0), vertice.squeeze(0), chunk_size_mri=200, stride_mri=100)
            for audio_chunk, vertice_chunk in zip(audio_chunks, vertice_chunks):
                audio_batch = audio_chunk.unsqueeze(0).to(device="cuda")
                vertice_batch = vertice_chunk.unsqueeze(0).to(device="cuda")
                template_batch = template.to(device="cuda")
                one_hot_all_batch = one_hot_all.to(device="cuda")
                train_subject = file_name[0].split("_")[0]
                if train_subject in train_subjects_list:
                    iter = train_subjects_list.index(train_subject)
                    one_hot = one_hot_all_batch[:, iter, :]
                    loss = model(audio_batch, template_batch, vertice_batch, one_hot, criterion)
                    valid_loss_log.append(loss.item())
                else:
                    for iter in range(one_hot_all_batch.shape[-1]):
                        one_hot = one_hot_all_batch[:, iter, :]
                        loss = model(audio_batch, template_batch, vertice_batch, one_hot, criterion)
                        valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)

        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print("epcoh: {}, current loss:{:.7f}".format(e+1, current_loss))
    return model

@torch.no_grad()
def test(args, model, test_loader, epoch):
    result_path = os.path.join(args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    save_path = os.path.join(args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        audio_chunks, vertice_chunks = split_into_chunks(audio.squeeze(0), vertice.squeeze(0), chunk_size_mri= 200, stride_mri=100)
        all_predictions = []
        chunk_index = 0
        for audio_chunk, vertice_chunk in zip(audio_chunks, vertice_chunks):
            chunk_index = chunk_index + 1
            audio_batch = audio_chunk.unsqueeze(0).to(device="cuda")
            template_batch = template.to(device="cuda")
            one_hot_all_batch = one_hot_all.to(device="cuda")
            train_subject = file_name[0].split("_")[0]
            if train_subject in train_subjects_list:
                iter = train_subjects_list.index(train_subject)
                one_hot = one_hot_all_batch[:, iter, :]
                prediction = model.predict(audio_batch, template_batch, one_hot)
                print(f"Chunk {chunk_index} prediction shape: {prediction.shape}")

                all_predictions.append(prediction.squeeze(0).detach().cpu().numpy())
            else:
                for iter in range(one_hot_all_batch.shape[-1]):
                    one_hot = one_hot_all_batch[:, iter, :]
                    prediction = model.predict(audio_batch, template_batch, one_hot)
                    all_predictions.append(prediction.squeeze(0).detach().cpu().numpy())

        full_prediction = np.concatenate(all_predictions, axis=0)
        np.save(os.path.join(result_path, file_name[0].split(".")[0]+"_prediction.npy"), full_prediction)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    
# if __name__ == "__main__":
import types

from transformers import Wav2Vec2Config

# Explicitly force eager attention and enable output_attentions
config = Wav2Vec2Config.from_pretrained(
    "facebook/wav2vec2-base-960h",
    attn_implementation="eager",         # 🚫 Avoids SDPA issue
    output_attentions=True,              # ✅ Safe now with eager
    output_hidden_states=False
)

# Create model with config
model = Faceformer(args, config=config)


# build model
#model = Faceformer(args)
print("model parameters: ", count_parameters(model))

model = model.to(torch.device("cuda"))

dataset = get_dataloaders(args)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)
test(args, model, dataset["test"], epoch=args.max_epoch)   