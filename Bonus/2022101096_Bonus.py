### **Importing Libraries**

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import string


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""### **Importing and Cleaning Dataset**"""

with open('./sample_data/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

print("Dataset Loaded")

corpus = corpus.lower()
clean_text = sent_tokenize(corpus)
translator = str.maketrans('', '', string.punctuation)
clean_text = [sentence.translate(translator) for sentence in clean_text]
# print(len(clean_text))

"""### **Tokenization and Emmbedding**"""

tokenized_corpus = [word_tokenize(sentence) for sentence in clean_text]
word_to_ind = {}
for i in range(len(tokenized_corpus)):
    token_arr = tokenized_corpus[i]

    #Vocabulary
    for tokken in token_arr:
        if tokken not in word_to_ind:
            word_to_ind[tokken] = len(word_to_ind)

    token_arr = ['<sos>'] * 5 + token_arr + ['<eos>'] * 5
    tokenized_corpus[i] = token_arr

# print(tokenized_corpus[2])
word_to_ind["<sos>"] = len(word_to_ind)
word_to_ind["<eos>"] = len(word_to_ind)
print(len(word_to_ind))

"""### **Test-Train Split**"""

train_val_data, test_data = train_test_split(tokenized_corpus, test_size=0.2)

train_data, validation_data = train_test_split(train_val_data, test_size=0.125)

# Print the sizes of each set
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")

"""### **Neural Network Model**"""

class NeuralLM(nn.Module):
    def __init__(self, emb_dim, hidden_size, context_size, vocab_size, pretrained_embeddings, dropout_rate):
        super(NeuralLM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings), freeze=True)
        #Model Layers
        self.l1 = torch.nn.Linear(self.context_size * self.emb_dim, self.hidden_size)
        self.a1 = torch.nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.l2 = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inp):
        #Prepairing Embeddings
        inp = self.embeddings(inp)
        inp = inp.view(inp.size(1), -1)
        # print(inp.shape)

        inp = self.l1(inp)
        inp = self.a1(inp)
        inp = self.dropout(inp)
        inp = self.l2(inp)
        return inp

"""### **Creating Datasets**"""

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, context_indices, next_word_indices):
        self.context_indices = context_indices
        self.next_word_indices = next_word_indices

    def __len__(self):
        return len(self.next_word_indices)

    def __getitem__(self, index):
        return torch.tensor(self.context_indices[index]), torch.tensor(self.next_word_indices[index])

"""### **Creating Input**"""

N_Gram = 5

def process_sentences(sentences, word_to_index, context_size):
    def words_to_indices(words, word_to_index):
        return [word_to_index.get(word, 0) for word in words]  # Default to 0 if word not in vocab

    context_indices = []
    central_word_indices = []

    for sentence in sentences:
        word_indices = words_to_indices(sentence, word_to_index)

        for i in range(len(sentence) - context_size):
            context_window = word_indices[i:i + context_size]
            context_indices.append(context_window)
            central_word_indices.append(word_to_index.get(sentence[i + context_size], 0))

    return context_indices, central_word_indices


train_gram_inp, train_cen_inp = process_sentences(train_data, word_to_ind, N_Gram)
val_gram_inp, val_cen_inp = process_sentences(validation_data, word_to_ind, N_Gram)
test_gram_inp, test_cen_inp = process_sentences(test_data, word_to_ind, N_Gram)

print(len(train_cen_inp))

# Function to train and validate the model
def train_and_evaluate(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader_train:
            context_words, target_words = batch
            # Move data to GPU
            trans_context = torch.transpose(context_words, 0, 1).to(device)
            target_words = target_words.to(device)

            outputs = model(trans_context)
            loss = criterion(outputs, target_words)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader_val:
                context_words, target_words = batch
                # Move data to GPU
                trans_context = torch.transpose(context_words, 0, 1).to(device)
                target_words = target_words.to(device)

                outputs = model(trans_context)
                loss = criterion(outputs, target_words)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)

    return train_losses, val_losses

# Function to test the model
def test_model(model, dataloader_test, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader_test:
            context_words, target_words = batch
            # Move data to GPU
            trans_context = torch.transpose(context_words, 0, 1).to(device)
            target_words = target_words.to(device)

            outputs = model(trans_context)
            loss = criterion(outputs, target_words)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += target_words.size(0)
            correct += (predicted == target_words).sum().item()

    accuracy = 100 * correct / total
    perplexity = math.exp(total_loss / len(dataloader_test))

    return accuracy, perplexity

# Hyperparameter variations
combinations = [
    (0.1, 100, 'Adam', 0.001),
    (0.3, 200, 'Adam', 0.001),
    (0.1, 300, 'SGD', 0.001),
    (0.3, 100, 'SGD', 0.01),
    (0.5, 200, 'Adam', 0.001)
]

criterion = nn.CrossEntropyLoss()
num_epochs = 5

# Store results for plotting
train_perplexities = []
val_perplexities = []
test_perplexities = []

dataset_train = EntityDataset(train_gram_inp, train_cen_inp)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128)

dataset_val = EntityDataset(val_gram_inp, val_cen_inp)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128)

dataset_test = EntityDataset(test_gram_inp, test_cen_inp)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128)

with open("perplexity_report.txt", "w") as f:
    for (dropout, dim, opt, lr) in combinations:
        print(f"Training with Dropout={dropout}, Dimension={dim}, Optimizer={opt}, Learning Rate={lr}")
        f.write(f"Training with Dropout={dropout}, Dimension={dim}, Optimizer={opt}, Learning Rate={lr}\n")

        word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=dim, window=5, min_count=1, workers=4)
        pretrained_embeddings = word2vec_model.wv.vectors

        model = NeuralLM(dim, 300, N_Gram, len(word_to_ind), pretrained_embeddings, dropout_rate=dropout)
        model.to(device)

        if opt == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)

        train_losses, val_losses = train_and_evaluate(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs)

        train_perplexities.append([math.exp(loss) for loss in train_losses])
        val_perplexities.append([math.exp(loss) for loss in val_losses])

        test_acc, test_perplexity = test_model(model, dataloader_test, criterion)
        test_perplexities.append(test_perplexity)

        f.write(f"Final Test Perplexity: {test_perplexity}\n")
        f.write("="*50 + "\n")

epochs = list(range(1, num_epochs + 1))
for i, (dropout, dim, opt, lr) in enumerate(combinations):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_perplexities[i], label='Train Perplexity')
    plt.plot(epochs, val_perplexities[i], label='Val Perplexity')
    plt.title(f'Dropout={dropout}, Dimension={dim}, Optimizer={opt}, Learning Rate={lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    # plt.show()
    plt.savefig(f'./plot_{i}.png')