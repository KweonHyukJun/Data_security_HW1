import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Set CUDA device order and visibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Makes GPU 0 and 1 available for use

# Improved device selection logic
if torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

MODEL_PATH = "spam_classifier_model.pt"  # Path to save the model

# Dataset class for loading the CSV
class SpamDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, label = self.texts[idx], self.labels[idx]
        encoded = [self.vocab.get(word, 0) for word in text.split()]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

def pad_collate(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts, labels, lengths

# Load and preprocess the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=["label", "text"])
    df['label'] = LabelEncoder().fit_transform(df['label'])  # ham -> 0, spam -> 1
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels

# Build vocabulary
def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.items())}  # 0 for padding
    return vocab

# Define the neural network
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=50):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

# Train the model
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.to(device)  # Move model to selected device
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(texts, lengths).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.6f}')
    print("Training complete.")

# Evaluate the model with F1 score and classification report
def evaluate(model, dataloader):
    model.to(device)  # Move model to device
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device)  # Move data to device
            outputs = model(texts, lengths).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["ham", "spam"])
    cm = confusion_matrix(all_labels, all_preds)

    print(f'Accuracy: {acc * 100:.2f}%')
    print(f'F1 Score: {f1:.4f}')
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

# Save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")

# Load a model from a .pt file
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}.")

# Main code
if __name__ == '__main__':
    texts, labels = load_data('spam.csv')
    vocab = build_vocab(texts)
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_dataset = SpamDataset(train_texts, train_labels, vocab)
    test_dataset = SpamDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate)

    model = TextClassifier(vocab_size=len(vocab) + 1)  # +1 for padding
    
    criterion = nn.SmoothL1Loss(beta=1.0)  # Use PyTorch's built-in loss function

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, epochs=50)
    save_model(model, MODEL_PATH)
    evaluate(model, test_loader)
