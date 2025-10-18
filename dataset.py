import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class EpicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, seq_len=10, participant="P01"):
        self.data = pd.read_csv(csv_file)

        
        self.data = self.data[self.data['participant_id'] == participant]

        
        existing_folders = {f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))}
        self.data = self.data[self.data['video_id'].isin(existing_folders)]

        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len
        self.data.reset_index(drop=True, inplace=True)

        print(f"Loaded {len(self.data)} samples for participant {participant}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row['video_id']
        start_frame = int(row['start_frame'])
        stop_frame = int(row['stop_frame'])
        verb_class = int(row['verb_class'])
        noun_class = int(row['noun_class'])

        folder = os.path.join(self.root_dir, video_id)

        
        frame_ids = list(range(start_frame, stop_frame))
        if len(frame_ids) == 0:
            frame_ids = [start_frame]
        sampled_frames = frame_ids[::max(1, len(frame_ids)//self.seq_len)][:self.seq_len]

        frames = []
        for fid in sampled_frames:
            frame_path = os.path.join(folder, f'frame_{fid:010d}.jpg')
            if not os.path.exists(frame_path):
                continue
            img = Image.open(frame_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        
        while len(frames) < self.seq_len:
            frames.append(torch.zeros_like(frames[0]))

        frames = torch.stack(frames)  # [T, 3, 224, 224]
        return frames, verb_class, noun_class