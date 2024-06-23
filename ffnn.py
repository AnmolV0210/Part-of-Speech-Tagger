import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import conllu
from collections import defaultdict

p = 2
s = 3

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return conllu.parse(data)

def extract_data(dataset):
    words = []
    tags = []
    for sentence in dataset:
        for token in sentence:
            words.append(token["form"].lower())
            tags.append(token["upos"])
    return words, tags

class PosDataset(Dataset):
    def __init__(self, word_indices, tag_indices, p, s):
        self.word_indices = word_indices
        self.tag_indices = tag_indices
        self.p = p
        self.s = s

    def __len__(self):
        return len(self.word_indices)

    def __getitem__(self, index):
        start_index = index - self.p
        if start_index < 0:
            extraprv = abs(index - self.p)
            start_index = 0
        else:
            extraprv = 0

        end_index = index + self.s
        if end_index > len(self.word_indices):
            extraend = (end_index) - (len(self.word_indices))
            end_index = len(self.word_indices) + 1
        else:
            extraend = 0

        context = [0] * extraprv
        context += self.word_indices[start_index:index] + self.word_indices[index:end_index]
        context += [0] * extraend  # Padding to ensure fixed context length

        tag = self.tag_indices[index]

        return torch.tensor(context), torch.tensor(tag)
    

class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, p, s):
        super(FFNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.p = p
        self.s = s
        self.fc1 = nn.Linear(embedding_dim * (p + s), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        output = torch.relu(self.fc1(embedded))
        output = self.fc2(output)
        return output

def train_and_save_model(model, train_loader, val_loader, criterion, optimizer, model_save_path, num_epochs=5, early_stopping_patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            val_loss /= len(val_loader.dataset)
            val_accuracy = correct / total
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the model if it's the best so far
                torch.save(model.state_dict(), model_save_path)
                print("Model saved.")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered!")
                    return

def test(model, test_loader, tag_to_index, index_to_tag, words):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)

            predicted_tags = [index_to_tag[idx.item()] for idx in predicted]

            for word, tag in zip(words, predicted_tags):
                print(f"{word}\t{tag}")

def main():
    train_data = load_data("en_atis-ud-train.conllu")
    val_data = load_data("en_atis-ud-dev.conllu")
    test_data = load_data("en_atis-ud-test.conllu")

    train_words, train_tags = extract_data(train_data)
    val_words, val_tags = extract_data(val_data)
    test_words, test_tags = extract_data(test_data)

    word_to_index = defaultdict(lambda: len(word_to_index))
    tag_to_index = defaultdict(lambda: len(tag_to_index))
    word_to_index["<PAD>"] = 0
    tag_to_index["<PAD>"] = 0

    for word, tag in zip(train_words, train_tags):
        word_to_index[word]
        tag_to_index[tag]

    def words_tags_to_indices(words, tags, word_to_index, tag_to_index):
        word_indices = [word_to_index[word] for word in words]
        tag_indices = [tag_to_index[tag] for tag in tags]
        return word_indices, tag_indices

    train_word_indices, train_tag_indices = words_tags_to_indices(train_words, train_tags, word_to_index, tag_to_index)
    val_word_indices, val_tag_indices = words_tags_to_indices(val_words, val_tags, word_to_index, tag_to_index)
    test_word_indices, test_tag_indices = words_tags_to_indices(test_words, test_tags, word_to_index, tag_to_index)

    train_dataset = PosDataset(train_word_indices, train_tag_indices, p, s)
    val_dataset = PosDataset(val_word_indices, val_tag_indices, p, s)
    test_dataset = PosDataset(test_word_indices, test_tag_indices, p, s)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    vocab_size = len(word_to_index)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = len(tag_to_index)

    model = FFNN(vocab_size, embedding_dim, hidden_dim, output_dim, p, s)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}

    model_save_path = "ffnn_model.pt"
    train_and_save_model(model, train_loader, val_loader, criterion, optimizer, model_save_path, num_epochs=20, early_stopping_patience=3)

    sentence = input("Enter a sentence: ")
    words = sentence.lower().split()

    word_indices = [word_to_index.get(word, word_to_index["<PAD>"]) for word in words]
    tag_indices = [-1] * len(word_indices)
    test_dataset_try = PosDataset(word_indices, tag_indices, p, s)

    bsize = len(word_indices)
    test_loader_try = DataLoader(test_dataset_try, batch_size=bsize)

    index_to_tag = {index: tag for tag, index in tag_to_index.items()}
    test(model, test_loader_try, tag_to_index, index_to_tag, words)


if __name__ == "__main__":
    main()
