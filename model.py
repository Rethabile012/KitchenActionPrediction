
import torch.nn as nn
import torchvision.models as models

class VerbNounLSTM(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_verbs=97, num_nouns=300):
        super(VerbNounLSTM, self).__init__()

        # CNN backbone (ResNet)
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # remove classifier

        # Freeze CNN (optional at first)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # LSTM
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=2, batch_first=True)

        # Two heads
        self.fc_verb = nn.Linear(hidden_dim, num_verbs)
        self.fc_noun = nn.Linear(hidden_dim, num_nouns)

    def forward(self, x):
        # x: [batch, seq_len, 3, 224, 224]
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)

        # Extract CNN features
        features = self.cnn(x).view(batch_size, seq_len, -1)  # (B, T, 2048)

        # LSTM
        _, (h_n, _) = self.lstm(features)
        hidden = h_n[-1]  # last layer hidden state

        # Predictions
        verb_logits = self.fc_verb(hidden)
        noun_logits = self.fc_noun(hidden)
        return verb_logits, noun_logits
