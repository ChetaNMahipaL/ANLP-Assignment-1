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
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

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


word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

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

class NeuralLM(nn.Module):
    def __init__(self, emb_dim, hidden_size, context_size, vocab_size, pretrained_embeddings):
        super(NeuralLM, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(pretrained_embeddings), freeze=True)

        self.l1 = torch.nn.Linear(context_size * emb_dim, hidden_size)
        self.a1 = torch.nn.Tanh()
        self.l2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, inp):
        # Lookup embeddings using word indices
        inp = self.embeddings(inp)
        
        inp = inp.view(inp.size(1), -1) 
        # print(inp.shape)
        
        inp = self.l1(inp)
        inp = self.a1(inp)
        inp = self.l2(inp)
        return inp

#<------------------------------------------------DataLoader---------------------------------------------------------------------->

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, context_indices, next_word_indices):
        self.context_indices = context_indices
        self.next_word_indices = next_word_indices

    def __len__(self):
        return len(self.next_word_indices)
    
    def __getitem__(self, index):
        return torch.tensor(self.context_indices[index]), torch.tensor(self.next_word_indices[index])
   
#<----------------------------------------------Creating Input-------------------------------------------------------------------->

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

# print(len(train_cen_inp))
print("Created input for loading")

#<-----------------------------------------------Training and Validation------------------------------------------------------>

print("Training Begins")
dataset_train = EntityDataset(train_gram_inp, train_cen_inp)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128)

dataset_val = EntityDataset(val_gram_inp, val_cen_inp)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128)

pretrained_embeddings = word2vec_model.wv.vectors

model = NeuralLM(100, 300, N_Gram, len(word_to_ind), pretrained_embeddings)
model.to(device)

num_epochs = 5
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader_train:
        context_words, target_words = batch
        trans_context = torch.transpose(context_words,0,1)
        trans_context = trans_context.to(device)
        target_words = target_words.to(device)

        outputs = model(trans_context) 
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
            trans_context = torch.transpose(context_words,0,1)
            trans_context = trans_context.to(device)
            target_words = target_words.to(device)
            
            outputs = model(trans_context)
            loss = criterion(outputs, target_words)
            total_val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += target_words.size(0)
            correct += (predicted == target_words).sum().item()
    
    avg_val_loss = total_val_loss / len(dataloader_val)
    accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')

print("Training and Validation Complete.")

#<-----------------------------------------------------Testing-------------------------------------------------------------->

print("Testing Begins")
dataset_test = EntityDataset(test_gram_inp, test_cen_inp)
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
        trans_context = torch.transpose(context_words,0,1)
        trans_context = trans_context.to(device)
        target_words = target_words.to(device)

        outputs = model(trans_context)
        # print(outputs)
        
        loss = criteria(outputs, target_words)
        total_loss += loss.item()
        # total_tokens += target_words.numel()
        
        _, predicted = torch.max(outputs, 1)
        total += target_words.size(0)
        correct += (predicted == target_words).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Perplexity is {math.exp(total_loss/len(dataloader_test))}')
