#<-----------------------------------------Importing Libraries------------------------------------------------->

import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
# nltk.download('punkt')
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import numpy as np
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#<-----------------------------------------Importing and Cleaning Dataset---------------------------------------------------->

with open('./datasets/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

corpus = corpus.lower()
clean_text = sent_tokenize(corpus)
print(len(clean_text))
filtered_corpus = [line for line in clean_text if line.strip()]


#<---------------------------------------------Tokenization and Emmbedding----------------------------------------------------->


tokenized_corpus = [word_tokenize(sentence) for sentence in filtered_corpus]
uniq_words = {}
for i in range(len(tokenized_corpus)):
    token_arr = tokenized_corpus[i]
    
    #Vocabulary
    for tokken in token_arr:
        if tokken not in uniq_words:
            uniq_words[tokken] = len(uniq_words)
    
    token_arr = ['<sos>'] * 5 + token_arr + ['<eos>'] * 5
    tokenized_corpus[i] = token_arr

# print(tokenized_corpus[2])
uniq_words["<sos>"] = len(uniq_words)
uniq_words["<eos>"] = len(uniq_words)
print(len(uniq_words))


word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=300, window=5, min_count=1, workers=4)

# sos_vector = word2vec_model.wv['<sos>']
# print(sos_vector)

# similarity = word2vec_model.wv.similarity('revolution', 'freedom')
# print(similarity)

# Get the entire embedding matrix
# embedding_matrix = word2vec_model.wv.vectors
# print("Embedding matrix shape:", embedding_matrix.shape)

# Find the word in the vocabulary that is closest to this vector
# most_similar_word = word2vec_model.wv.similar_by_vector(embedding_matrix[263], topn=1)
# print(most_similar_word[0][0])


word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=300, window=5, min_count=1, workers=4)

sos_vector = word2vec_model.wv['<sos>']
print(sos_vector)

# similarity = word2vec_model.wv.similarity('revolution', 'freedom')
# print(similarity)

# Get the entire embedding matrix
# embedding_matrix = word2vec_model.wv.vectors
# print("Embedding matrix shape:", embedding_matrix.shape)

# Find the word in the vocabulary that is closest to this vector
# most_similar_word = word2vec_model.wv.similar_by_vector(embedding_matrix[263], topn=1)
# print(most_similar_word[0][0])

#<-------------------------------------------------Test-Train Split-------------------------------------------------------------->


train_val_data, test_data = train_test_split(tokenized_corpus, test_size=int(0.2*(len(tokenized_corpus))), random_state=42)

# Then, split the remaining data into training and validation sets
train_data, validation_data = train_test_split(train_val_data, test_size=int(0.1*(len(tokenized_corpus))), random_state=42)

# Print the sizes of each set
print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(validation_data)}")
print(f"Test data size: {len(test_data)}")



#<-------------------------------------------------Neural Network Model---------------------------------------------------------->

class NeuralLM(nn.Module):
    def __init__(self,emb_dim,hidden_size, context_size, vocab_size):
        super(NeuralLM,self).__init__()
        self.l1 = torch.nn.Linear(context_size*emb_dim, hidden_size)
        self.a1 = torch.nn.Tanh()
        self.l2 = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, inp):
        inp = inp.to(device)  # Ensure input is on the same device as the model
        inp = self.l1(inp)
        inp = self.a1(inp)
        inp = self.l2(inp)
        return inp
    
#<------------------------------------------------DataLoader---------------------------------------------------------------------->

class EntityDataset(torch.utils.data.Dataset):
    def __init__(self, concatenatedEmbeddings, nextWordIndices):
        self.concatenatedEmbeddings = concatenatedEmbeddings
        self.nextWordIndices = nextWordIndices

    def __len__(self):
        return len(self.nextWordIndices)
    
    def __getitem__(self, index):
        return torch.tensor(self.concatenatedEmbeddings[index]), torch.tensor(self.nextWordIndices[index])
    
#<----------------------------------------------Creating Input-------------------------------------------------------------------->

N_Gram = 5

def process_sentences(sentences, w2v_model, context_size):

    embedding_matrix = w2v_model.wv.vectors
    word_to_index = {word: idx for idx, word in enumerate(w2v_model.wv.index_to_key)}

    def words_to_indices(words, word_to_index):
        return [word_to_index.get(word, 0) for word in words]  # Default to 0 if word not in vocab

    concatenated_contexts = []
    central_word = []

    for sentence in sentences:
        word_indices = words_to_indices(sentence, word_to_index)

        embeddings = embedding_matrix[word_indices]

        for i in range(len(sentence) - context_size):
            context_window = embeddings[i:i + context_size]
            concatenated_context = context_window.flatten()  
            concatenated_contexts.append(concatenated_context)
            central_word.append(uniq_words[sentence[i + context_size]])

    concatenated_contexts = np.array(concatenated_contexts)
    return concatenated_contexts, central_word

train_gram_inp, train_cen_inp = process_sentences(train_data,word2vec_model,N_Gram)
print(len(train_cen_inp))

val_gram_inp, val_cen_inp = process_sentences(validation_data,word2vec_model,N_Gram)
test_gram_inp, test_cen_inp = process_sentences(test_data,word2vec_model,N_Gram)

#<-----------------------------------------------Training and Validation------------------------------------------------------>

dataset_train = EntityDataset(train_gram_inp, train_cen_inp)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=128)


dataset_val = EntityDataset(val_gram_inp, val_cen_inp)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=128)

# Assuming NeuralLM is your FFNNPredictor or any other model that you've defined
model = NeuralLM(300, 300, 5, len(uniq_words))  # Ensure inputSize matches concatenated embedding size
num_epochs = 5
learning_rate = 0.001
model.to(device)  # Move the model to GPU

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    total_loss = 0
    for batch in dataloader_train:
        concatenated_embeds, target_words = batch

        concatenated_embeds = concatenated_embeds.to(device)
        target_words = target_words.to(device)

        
        optimizer.zero_grad()  # Clear the gradients
        
        outputs = model(concatenated_embeds)  # Forward pass
        
        loss = criterion(outputs, target_words)  # Compute loss
        
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters
        
        total_loss += loss.item()  # Accumulate loss
    
    avg_train_loss = total_loss / len(dataloader_train)
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader_val:
            concatenated_embeds, target_words = batch
            concatenated_embeds = concatenated_embeds.to(device)
            target_words = target_words.to(device)
            
            outputs = model(concatenated_embeds)  # Forward pass
            
            # Compute loss
            loss = criterion(outputs, target_words)
            total_val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += target_words.size(0)
            correct += (predicted == target_words).sum().item()
    
    avg_val_loss = total_val_loss / len(dataloader_val)
    accuracy = 100 * correct / total
    
    # Print metrics
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

print("Training and validation complete.")

#<-----------------------------------------------------Testing-------------------------------------------------------------->

dataset_test = EntityDataset(test_gram_inp, test_cen_inp)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=128)

model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for evaluation
    for batch in dataloader_test:
        concatenated_embeds, target_words = batch
        
        # Move data to GPU
        concatenated_embeds = concatenated_embeds.to(device)
        target_words = target_words.to(device)
        
        # Forward pass
        outputs = model(concatenated_embeds)
        
        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        
        # Calculate number of correct predictions
        total += target_words.size(0)
        correct += (predicted == target_words).sum().item()

accuracy = 100 * correct / total
print(accuracy)
