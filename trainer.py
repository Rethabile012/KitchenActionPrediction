import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


CSV_PATH = '/kaggle/input/kitchen/Dataset/EPIC_100_train.csv'
FRAME_ROOT = '/kaggle/working/frames'
SAVE_MODEL_PATH = 'best_lstm_action_model.pth'
LOSS_LOG_PATH = 'training_loss.csv'

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4  
SEQUENCE_LENGTH = 16
IMG_SIZE = 224



class EpicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, seq_len=10, participant="P01"):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['participant_id'] == participant]

        # Only keep rows with valid video folders
        existing_folders = {f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))}
        self.data = self.data[self.data['video_id'].isin(existing_folders)]
        self.data.reset_index(drop=True, inplace=True)

        self.root_dir = root_dir
        self.transform = transform
        self.seq_len = seq_len

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
            try:
                img = Image.open(frame_path).convert("RGB")
            except Exception:
                continue
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Handle missing or empty frames
        if len(frames) == 0:
            frames = [torch.zeros(3, IMG_SIZE, IMG_SIZE)] * self.seq_len

        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        frames = torch.stack(frames)
        return frames, verb_class, noun_class



class VerbNounLSTM(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_verbs=97, num_nouns=300):
        super(VerbNounLSTM, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer

        # Freeze all except the last layer block (layer4)
        for name, param in self.cnn.named_parameters():
            param.requires_grad = ("layer4" in name)

        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc_verb = nn.Linear(hidden_dim, num_verbs)
        self.fc_noun = nn.Linear(hidden_dim, num_nouns)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)
        _, (h_n, _) = self.lstm(features)
        h = h_n[-1]
        verb_logits = self.fc_verb(h)
        noun_logits = self.fc_noun(h)
        return verb_logits, noun_logits



def compute_metrics(verb_preds, noun_preds, verb_labels, noun_labels):
    """Computes Top-1, Top-5, and Action Pair Accuracy."""
    with torch.no_grad():
        verb_top1 = (verb_preds.argmax(dim=1) == verb_labels).float().mean().item()
        noun_top1 = (noun_preds.argmax(dim=1) == noun_labels).float().mean().item()

        verb_top5 = (verb_labels.view(-1, 1) == verb_preds.topk(5, dim=1).indices).any(dim=1).float().mean().item()
        noun_top5 = (noun_labels.view(-1, 1) == noun_preds.topk(5, dim=1).indices).any(dim=1).float().mean().item()

        correct_action = ((verb_preds.argmax(dim=1) == verb_labels) &
                          (noun_preds.argmax(dim=1) == noun_labels)).float().mean().item()

    return verb_top1, noun_top1, verb_top5, noun_top5, correct_action



def train_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = EpicDataset(CSV_PATH, FRAME_ROOT, transform)
    if len(full_dataset) == 0:
        raise ValueError("No valid samples found! Check that frames and CSV video_ids match.")

    val_split = 0.1
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VerbNounLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float('inf')
    loss_log = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for frames, verb_labels, noun_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)
            optimizer.zero_grad()
            verb_preds, noun_preds = model(frames)
            loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        metrics = {'verb_top1': 0, 'noun_top1': 0, 'verb_top5': 0, 'noun_top5': 0, 'action_pair': 0}
        count = 0

        with torch.no_grad():
            for frames, verb_labels, noun_labels in val_loader:
                frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)
                verb_preds, noun_preds = model(frames)
                loss = criterion(verb_preds, verb_labels) + criterion(noun_preds, noun_labels)
                total_val_loss += loss.item()

                v1, n1, v5, n5, a = compute_metrics(verb_preds, noun_preds, verb_labels, noun_labels)
                metrics['verb_top1'] += v1
                metrics['noun_top1'] += n1
                metrics['verb_top5'] += v5
                metrics['noun_top5'] += n5
                metrics['action_pair'] += a
                count += 1

        for k in metrics:
            metrics[k] /= count

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()

        print(f"\n Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Verb Top1: {metrics['verb_top1']:.4f} | Noun Top1: {metrics['noun_top1']:.4f}")
        print(f"Verb Top5: {metrics['verb_top5']:.4f} | Noun Top5: {metrics['noun_top5']:.4f}")
        print(f"Action Pair Accuracy: {metrics['action_pair']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"New best model saved (Val Loss: {avg_val_loss:.4f})")

        loss_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **metrics
        })

    pd.DataFrame(loss_log).to_csv(LOSS_LOG_PATH, index=False)
    print(f"Training complete. Best model saved to {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train_model()
