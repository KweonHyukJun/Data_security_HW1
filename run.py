import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from collections import Counter

MODEL_PATH = "spam_classifier_model.pt"  # Path to the saved model
TEST_FILE = "test.csv"  # Test data CSV file
RESULT_FILE = "result.txt"  # Output file for results

# Dataset class for test data
class TestDataset(Dataset):
    def __init__(self, texts, vocab):
        self.texts = texts
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = [self.vocab.get(word, 0) for word in text.split()]
        return torch.tensor(encoded, dtype=torch.long)

def pad_collate(batch):
    texts = batch
    lengths = torch.tensor([len(text) for text in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, lengths

# Load vocabulary from the training phase
def load_vocab(file_path):
    df = pd.read_csv(file_path, header=None, names=["label", "text"])
    counter = Counter()
    for text in df['text'].tolist():
        counter.update(text.split())
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(counter.items())}  # 0 for padding
    return vocab

# Define the neural network (same as used in training)
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

# Load the model from the .pt file
def load_model(model_path, vocab_size):
    model = TextClassifier(vocab_size=vocab_size + 1)  # +1 for padding
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Perform inference on the test data
def inference(model, dataloader):
    results = []
    with torch.no_grad():
        for texts, lengths in dataloader:
            outputs = model(texts, lengths).squeeze()
            preds = (outputs > 0.5).long().tolist()
            results.extend(preds)
    return results

# Save the inference results to a file
def save_results(results, output_file):
    with open(output_file, "w") as f:
        for result in results:
            # label = "spam" if result == 1 else "ham"
            f.write(f"{result}\n")
    print(f"Results saved to {output_file}.")

# Main function to load model, perform inference, and save results
def main():
    # Load vocabulary
    vocab = load_vocab('spam.csv')  # Assumes the same training dataset

    # Load test data, skipping the first column (labels)
    df = pd.read_csv(TEST_FILE, header=None, names=["label", "text"])
    test_texts = df["text"].tolist()

    # Create DataLoader for the test dataset
    test_dataset = TestDataset(test_texts, vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=pad_collate)

    # Load the trained model
    model = load_model(MODEL_PATH, vocab_size=len(vocab))

    # Perform inference
    results = inference(model, test_loader)

    # Save the results to a text file
    save_results(results, RESULT_FILE)

if __name__ == "__main__":
    main()
