import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import classification_report

# Set up environment for CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Check if CUDA is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CSV file into a pandas DataFrame
csv_file_path = './spam.csv'  # Use the uploaded file path
data = pd.read_csv(csv_file_path)

# Preview the loaded data
print("Data preview:\n", data.head())

# Process the data by converting labels to numerical values
data = data[['v1', 'v2']].dropna()  # Drop rows with missing values
data['label'] = data['v1'].apply(lambda x: 0 if x == 'ham' else 1)  # Convert labels to 0 and 1
data = data[['label', 'v2']]  # Keep only labels and text

# Print processed data
# print("Processed data:\n", data.head())

# Tokenizer and Vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for text in data_iter['v2']:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(data))

# Ensure unknown tokens are mapped to index 0 (if missing)
UNK_IDX = vocab.get("<unk>", 0)  # Use index 0 for unknown tokens


# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, text = self.data.iloc[idx]['label'], self.data.iloc[idx]['v2']
        text_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(text)], dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return text_tensor, label_tensor

# Collate function for batching
def collate_batch(batch, vocab):
    texts, labels = [], []
    for text, label in batch:
        texts.append(text)
        labels.append(label)
    texts = nn.utils.rnn.pad_sequence(texts, padding_value=vocab["<unk>"])
    labels = torch.tensor(labels, dtype=torch.float)
    return texts, labels

# Create Dataset and DataLoader
train_dataset = TextDataset(data, vocab, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: collate_batch(x, vocab))

# RNN-based Text Classification Model
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text):
        embedded = self.embedding(text).permute(1, 0, 2)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Initialize the model, loss function, and optimizer
model = TextClassificationModel(vocab_size=len(vocab), embed_size=64, hidden_size=32, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze(1)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=["ham", "spam"]))

# Train the model
train(model, train_loader, criterion, optimizer, device, epochs=5)

# Save the model
model_path = 'text_classification_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load and evaluate the model
loaded_model = TextClassificationModel(vocab_size=len(vocab), embed_size=64, hidden_size=32, output_size=1).to(device)
loaded_model.load_state_dict(torch.load(model_path))
print("Model loaded from disk")
evaluate(loaded_model, train_loader, device)
