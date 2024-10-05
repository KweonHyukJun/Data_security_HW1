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

# Assume the CSV columns are named 'v1' for labels and 'v2' for text content
# We need to convert labels (e.g., 'ham' and 'spam') into numerical labels (0 and 1)
data = data[['v1', 'v2']].dropna()  # Drop any rows with missing values
data['label'] = data['v1'].apply(lambda x: 0 if x == 'ham' else 1)  # Convert labels to 0 and 1
data = data[['label', 'v2']]  # Keep only the numerical labels and text data

# Print the processed data
print("Processed data:\n", data.head())

# Tokenizer and Vocabulary
tokenizer = get_tokenizer("basic_english")

# Define a function to yield tokenized text
def yield_tokens(data_iter):
    for text in data_iter['v2']:  # Iterate over the text column
        yield tokenizer(text)

# Build vocabulary from the text data
vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Define a custom Dataset
class TextDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label, text = self.data.iloc[idx]['label'], self.data.iloc[idx]['v2']
        # Tokenize and numericalize text
        text_tensor = torch.tensor([self.vocab[token] for token in self.tokenizer(text)], dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return text_tensor, label_tensor

# Define a function to pad and batch the data
def collate_batch(batch, vocab):
    texts, labels = [], []
    for (text, label) in batch:
        texts.append(text)
        labels.append(label)
    # Pad sequences to the maximum length in batch
    texts = nn.utils.rnn.pad_sequence(texts, padding_value=vocab["<unk>"])
    labels = torch.tensor(labels, dtype=torch.float)
    return texts, labels

# Create the Dataset and DataLoader for training
train_dataset = TextDataset(data, vocab, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: collate_batch(x, vocab))

# Define a simple RNN-based model for text classification
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, text):
        embedded = self.embedding(text).permute(1, 0, 2)  # Permute to (batch, seq, embed)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Define model, loss function, and optimizer
model = TextClassificationModel(vocab_size=len(vocab), embed_size=64, hidden_size=32, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs.squeeze(1), labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
    



# Train the model
train(model, train_loader, criterion, optimizer, device, epochs=5)

# Save the model to a .pt file
model_path = 'text_classification_model.pt'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Load the model for evaluation
loaded_model = TextClassificationModel(vocab_size=len(vocab), embed_size=64, hidden_size=32, output_size=1).to(device)
loaded_model.load_state_dict(torch.load(model_path))
print("Model loaded from disk")

# Evaluate the loaded model
evaluate(loaded_model, train_loader, device)
