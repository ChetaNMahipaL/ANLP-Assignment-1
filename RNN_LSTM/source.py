#<-----------------------------------------Importing Libraries------------------------------------------------->

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import nltk
# nltk.download('punkt')
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#<-----------------------------------------Importing and Cleaning Dataset---------------------------------------------------->

with open('./datasets/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

print("Dataset Loaded")

corpus = corpus.lower()
clean_text = sent_tokenize(corpus)
# print(len(clean_text))


#<---------------------------------------------Tokenization and Emmbedding----------------------------------------------------->

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
# print(len(word_to_ind))
print("Tokanized the input")


word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=200, window=5, min_count=1, workers=4)

print("Prepare Word Embeddings")

#<-------------------------------------------------Test-Train Split-------------------------------------------------------------->

print("Splitting Data")
train_val_data, test_data = train_test_split(tokenized_corpus, test_size=0.2)

train_data, validation_data = train_test_split(train_val_data, test_size=0.125)

# Print the sizes of each set
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")




#<-------------------------------------------------Neural Network Model---------------------------------------------------------->

class NeuralLM(nn.Module): #https://cnvrg.io/pytorch-lstm/
    def __init__(self, emb_dim, hidden_size, vocab_size, pretrained_embeddings, num_layers=1):
        super(NeuralLM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings), freeze=True)
        
        # LSTM layer
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.act_fn = nn.ReLU()
        self.dense_layer = nn.Linear(hidden_size, 128) #https://stackoverflow.com/questions/61149523/understanding-the-structure-of-my-lstm-model
        # Output layer
        self.class_layer = nn.Linear(128, vocab_size)

    def forward(self, inp):
        
        embedded = self.embeddings(inp)
        
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        # Apply dense layer and activation
        dense_out = self.act_fn(self.dense_layer(self.act_fn(lstm_out)))
        # Final output layer
        logits = self.class_layer(dense_out)
        
        return logits

#<------------------------------------------------DataLoader---------------------------------------------------------------------->

class LM_Dataset(torch.utils.data.Dataset):
    def __init__(self, sentences, targets):
        self.sentences = sentences
        self.targets = targets

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        target = self.targets[idx]
        return torch.tensor(sentence), torch.tensor(target)
   
#<----------------------------------------------Creating Input-------------------------------------------------------------------->

N_Gram = 5

def process_sentences(sentences, word_to_index, max_len=None):
    def words_to_indices(words, word_to_index):
        return [word_to_index.get(word, 0) for word in words]
    
    context_indices = []
    central_word_indices = []

    for sentence in sentences:
        word_indices = words_to_indices(sentence, word_to_index)
        
        if max_len is not None:
            word_indices = word_indices[:max_len] + [0] * (max_len - len(word_indices))

        context_indices.append(word_indices[:-1])
        
        central_word_indices.append(word_indices[1:])

    return context_indices, central_word_indices

train_gram_inp, train_cen_inp = process_sentences(train_data, word_to_ind, max_len=20)
val_gram_inp, val_cen_inp = process_sentences(validation_data, word_to_ind, max_len=20)
test_gram_inp, test_cen_inp = process_sentences(test_data, word_to_ind, max_len=20)

# print(len(train_cen_inp))
print("Created input for loading")

#<-----------------------------------------------Training and Validation------------------------------------------------------>

print("Training Begins")
dataset_train = LM_Dataset(train_gram_inp, train_cen_inp)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=True)

dataset_val = LM_Dataset(val_gram_inp, val_cen_inp)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128)

pretrained_embeddings = word2vec_model.wv.vectors

model = NeuralLM(200, 300, len(word_to_ind), pretrained_embeddings, num_layers=1)
model.to(device)

num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in dataloader_train:
        context_words, target_words = batch
        context_words = context_words.to(device)
        target_words = target_words.to(device)

        outputs = model(context_words)  
        outputs = outputs.view(-1, outputs.size(-1))
        target_words = target_words.view(-1)
        
        loss = criterion(outputs, target_words)
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad() 
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(dataloader_train)

    # Validation loop
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader_val:
            context_words, target_words = batch
            context_words = context_words.to(device)
            target_words = target_words.to(device)
            
            outputs = model(context_words)
            outputs = outputs.view(-1, outputs.size(-1))
            target_words = target_words.view(-1)
            
            loss = criterion(outputs, target_words)
            total_val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)  
            total += target_words.size(0)
            correct += (predicted == target_words).sum().item()
    
    avg_val_loss = total_val_loss / len(dataloader_val)
    accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

print("Training and Validation Complete.")

#<-----------------------------------------------------Testing-------------------------------------------------------------->

print("Testing Begins")
dataset_test = LM_Dataset(test_gram_inp, test_cen_inp)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128)

model.eval()
correct = 0
total = 0
total_loss = 0
total_tokens = 0
criteria = nn.CrossEntropyLoss()


with torch.no_grad():
    for batch in dataloader_test:
        context_words, target_words = batch
        context_words = context_words.to(device)
        target_words = target_words.to(device)

        outputs = model(context_words)  
        outputs = outputs.view(-1, outputs.size(-1))
        target_words = target_words.view(-1)
        
        loss = criterion(outputs, target_words)
        total_loss += loss.item()
        # total_tokens += target_words.numel()
        
        _, predicted = torch.max(outputs, 1)
        total += target_words.size(0)
        correct += (predicted == target_words).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
print(math.exp(total_loss/len(dataloader_test)))

