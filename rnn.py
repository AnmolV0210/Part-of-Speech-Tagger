import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import conllu
from collections import defaultdict

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return conllu.parse(data)

def extract_data(dataset):
    words = []
    tags = []
    for sentence in dataset:
        word_seq = []
        tag_seq = []
        for token in sentence:
            word_seq.append(token["form"].lower())
            tag_seq.append(token["upos"])
        words.append(word_seq)
        tags.append(tag_seq)
    return words, tags

def create_vocabulary(words, tags):
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    tag_to_idx = defaultdict(lambda: len(tag_to_idx))

    word_to_idx['<PAD>'] = 0
    tag_to_idx['<PAD>'] = 0

    for sentence_words, sentence_tags in zip(words, tags):
        for word in sentence_words:
            word_to_idx[word]
        for tag in sentence_tags:
            tag_to_idx[tag]

    return word_to_idx, tag_to_idx

def convert_to_indices(words, tags, word_to_idx, tag_to_idx):
    word_indices = [[word_to_idx[word] for word in sentence] for sentence in words]
    tag_indices = [[tag_to_idx[tag] for tag in sentence] for sentence in tags]
    return word_indices, tag_indices

# Pad sequences
def pad_sequences(sequences, pad_token=0):
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        padded_seq = seq + [pad_token] * (max_len - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences

class POSTaggingDataset(Dataset):
    def __init__(self, word_indices, tag_indices):
        self.word_indices = word_indices
        self.tag_indices = tag_indices

    def __len__(self):
        return len(self.word_indices)

    def __getitem__(self, idx):
        return torch.tensor(self.word_indices[idx]), torch.tensor(self.tag_indices[idx])

class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=-1)
        return tag_scores

def train_with_early_stopping(model, train_dataloader, val_dataloader, optimizer, loss_function, tagset_size, patience=5, epochs=100):
    best_val_loss = float('inf')
    no_improvement = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, (inputs, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            tag_scores = model(inputs)
            loss = loss_function(tag_scores.view(-1, tagset_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {average_train_loss}")

        val_loss = evaluate(model, val_dataloader, loss_function, tagset_size)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

        val_accuracy = evaluate_accuracy(model, val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_accuracy}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == patience:
                print(f"No improvement in validation loss for {patience} epochs. Early stopping")
                break

def evaluate(model, dataloader, loss_function, tagset_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            tag_scores = model(inputs)
            loss = loss_function(tag_scores.view(-1, tagset_size), targets.view(-1))
            total_loss += loss.item()
            total_tokens += targets.numel()
    return total_loss / total_tokens

def tag_sentence(model, sentence, word_to_idx, tag_to_idx):
    words = sentence.lower().split()
    idx_to_tag = {index: tag for tag, index in tag_to_idx.items()}
    word_indices = [word_to_idx.get(word, word_to_idx["<PAD>"]) for word in words]

    dataset = POSTaggingDataset([word_indices], [[-1] * len(word_indices)])  # Use -1 as placeholder for tags
    dataloader = DataLoader(dataset, batch_size=1)
    model.eval()

    with torch.no_grad():
        for inputs, _ in dataloader:
            tag_scores = model(inputs)
            _, predicted = torch.max(tag_scores, 2)
            predicted_tags = [idx_to_tag[idx.item()] for idx in predicted[0]]
    return list(zip(words, predicted_tags))


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            tag_scores = model(inputs)
            _, predicted = torch.max(tag_scores, 2)
            correct += (predicted == targets).sum().item()
            total += targets.numel()
    accuracy = correct / total
    return accuracy

def main():
    train_data = load_data("en_atis-ud-train.conllu")
    val_data = load_data("en_atis-ud-dev.conllu")
    test_data = load_data("en_atis-ud-test.conllu")

    train_words, train_tags = extract_data(train_data)
    val_words, val_tags = extract_data(val_data)
    test_words, test_tags = extract_data(test_data)

    word_to_idx, tag_to_idx = create_vocabulary(train_words, train_tags)

    train_word_indices, train_tag_indices = convert_to_indices(train_words, train_tags, word_to_idx, tag_to_idx)
    val_word_indices, val_tag_indices = convert_to_indices(val_words, val_tags, word_to_idx, tag_to_idx)
    test_word_indices, test_tag_indices = convert_to_indices(test_words, test_tags, word_to_idx, tag_to_idx)

    train_word_indices_padded = pad_sequences(train_word_indices)
    train_tag_indices_padded = pad_sequences(train_tag_indices)

    val_word_indices_padded = pad_sequences(val_word_indices)
    val_tag_indices_padded = pad_sequences(val_tag_indices)

    test_word_indices_padded = pad_sequences(test_word_indices)
    test_tag_indices_padded = pad_sequences(test_tag_indices)

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    VOCAB_SIZE = len(word_to_idx)
    TAGSET_SIZE = len(tag_to_idx)

    model = LSTMTagger(VOCAB_SIZE, TAGSET_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    idx_to_tag = {index: tag for tag, index in tag_to_idx.items()}

    train_dataset = POSTaggingDataset(train_word_indices_padded, train_tag_indices_padded)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = POSTaggingDataset(val_word_indices_padded, val_tag_indices_padded)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    test_dataset = POSTaggingDataset(test_word_indices_padded, test_tag_indices_padded)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    train_with_early_stopping(model, train_dataloader, val_dataloader, optimizer, loss_function, TAGSET_SIZE)

    torch.save(model.state_dict(), "rnn_model.pt")

    sentence = input("Enter a sentence: ")
    tagged_sentence = tag_sentence(model, sentence, word_to_idx, tag_to_idx)
    for word, tag in tagged_sentence:
        print(word + "  " + tag)


if __name__ == "__main__":
    main()