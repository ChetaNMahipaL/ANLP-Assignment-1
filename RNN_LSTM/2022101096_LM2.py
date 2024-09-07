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
import string
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#<-----------------------------------------Importing and Cleaning Dataset---------------------------------------------------->

with open('./datasets/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

print("Dataset Loaded")

corpus = corpus.lower()
clean_text = sent_tokenize(corpus)
translator = str.maketrans('', '', string.punctuation)
clean_text = [sentence.translate(translator) for sentence in clean_text]
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

train_gram_inp, train_cen_inp = process_sentences(train_data, word_to_ind, max_len=40)
val_gram_inp, val_cen_inp = process_sentences(validation_data, word_to_ind, max_len=40)
test_gram_inp, test_cen_inp = process_sentences(test_data, word_to_ind, max_len=40)

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
perp_vis_t = []
perp_vis_val = []
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

with open('2022101096_LM2_Train_Perplexity.txt', 'w') as train_file, open('2022101096_LM2_Validation_Perplexity.txt', 'w') as val_file:
     
    train_file.write(f'Epoch\tBatch\tPerplexity\n')
    val_file.write(f'Epoch\tBatch\tPerplexity\n')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_index, batch in enumerate(dataloader_train):
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
            batch_perplexity_t = math.exp(loss.item())
                    
            train_file.write(f'{epoch+1}\t{batch_index+1}\t{batch_perplexity_t:.4f}\n')

        avg_train_loss = total_loss / len(dataloader_train)
        train_perplexity = math.exp(avg_train_loss)
        
        # Validation loop
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader_val):
                context_words, target_words = batch
                context_words = context_words.to(device)
                target_words = target_words.to(device)
                
                outputs = model(context_words)
                outputs = outputs.view(-1, outputs.size(-1))
                target_words = target_words.view(-1)
                
                loss = criterion(outputs, target_words)
                total_val_loss += loss.item()
                batch_perplexity = math.exp(loss.item())
                    
                val_file.write(f'{epoch+1}\t{batch_index+1}\t{batch_perplexity:.4f}\n')
                
                _, predicted = torch.max(outputs, 1)  
                total += target_words.size(0)
                correct += (predicted == target_words).sum().item()
        
        avg_val_loss = total_val_loss / len(dataloader_val)
        val_perplexity = math.exp(avg_val_loss)
        accuracy = 100 * correct / total   

        perp_vis_t.append(train_perplexity)
        perp_vis_val.append(val_perplexity)
    
        train_file.write(f'End of Epoch {epoch+1} - Average Train Perplexity: {train_perplexity:.4f}\n')
        val_file.write(f'End of Epoch {epoch+1} - Average Val Perplexity: {val_perplexity:.4f}\n')

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}, Val Accuracy: {accuracy:.2f}%')
        
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

with open('2022101096_LM2_Test_Perplexity.txt', 'w') as f:
    total_loss = 0
    f.write(f'Batch\tPerplexity\n')

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader_test):
            context_words, target_words = batch
            context_words = context_words.to(device)
            target_words = target_words.to(device)

            outputs = model(context_words)
            outputs = outputs.view(-1, outputs.size(-1))
            target_words = target_words.view(-1)

            loss = criterion(outputs, target_words)
            total_loss += loss.item()

            perplexity = math.exp(loss.item())

            f.write(f'{batch_index+1}\t{perplexity:.4f}\n')

        average_perplexity = math.exp(total_loss / len(dataloader_test))
        f.write(f'Average perplexity:\t{average_perplexity:.4f}\n')

    print(f'Average Perplexity: {average_perplexity:.4f}')

#<-----------------------------------------------------------------Visualization------------------------------------------------>

epochs = range(1, len(perp_vis_t) + 1)  # Assuming each entry corresponds to an epoch

plt.figure(figsize=(10, 6))
plt.plot(epochs, perp_vis_t, marker='o', linestyle='-', color='blue', label='Training Perplexity')
plt.plot(epochs, perp_vis_val, marker='o', linestyle='--', color='red', label='Validation Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.title('Training and Validation Perplexity Over Epochs')
plt.legend()
plt.savefig('perplexity_plot.png')
# plt.show()

#<-------------------------------------------------------------------Save Model-------------------------------------------------->

# Save the entire model
torch.save(model, 'model/model_LM2.pth')