import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from dataset import EpicDataset
from model import VerbNounLSTM
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

CSV_PATH = 'EPIC_100_train.csv'   # path to your training CSV
FRAME_ROOT = 'frames'         # root folder containing folders like P01_101, P01_102...
SAVE_MODEL_PATH = 'lstm_action_model.pth'
LOSS_LOG_PATH = 'training_loss.csv'

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 16  # number of frames per clip
IMG_SIZE = 128


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def train_model():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    # Split dataset into train and validation
    full_dataset = EpicDataset(CSV_PATH, FRAME_ROOT, transform)
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

    best_val_loss = float('inf')
    loss_log = []

    for epoch in range(EPOCHS):
        # ---------- Training ----------
        model.train()
        total_train_loss = 0.0
        for frames, verb_labels, noun_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)

            optimizer.zero_grad()
            verb_preds, noun_preds = model(frames)
            loss_verb = criterion(verb_preds, verb_labels)
            loss_noun = criterion(noun_preds, noun_labels)
            loss = loss_verb + loss_noun
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---------- Validation ----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for frames, verb_labels, noun_labels in val_loader:
                frames, verb_labels, noun_labels = frames.to(device), verb_labels.to(device), noun_labels.to(device)
                verb_preds, noun_preds = model(frames)
                loss_verb = criterion(verb_preds, verb_labels)
                loss_noun = criterion(noun_preds, noun_labels)
                loss = loss_verb + loss_noun
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---------- Save Best Model ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"New best model saved at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})")

        # Log losses
        loss_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

    # Save loss log to CSV
    pd.DataFrame(loss_log).to_csv(LOSS_LOG_PATH, index=False)
    print(f"Training complete. Best model saved to {SAVE_MODEL_PATH}")

if __name__ == "__main__":
    train_model()