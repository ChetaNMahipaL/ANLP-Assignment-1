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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#<-----------------------------------------Importing and Cleaning Dataset---------------------------------------------------->

with open('./datasets/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

corpus = corpus.lower()
clean_text = sent_tokenize(corpus)
print(len(clean_text))
filtered_corpus = [line for line in clean_text if line.strip()]


#<---------------------------------------------Tokenization and Emmbedding----------------------------------------------------->


tokenized_corpus = [word_tokenize(sentence) for sentence in filtered_corpus]
uniq_words = []
for i in range(len(tokenized_corpus)):
    token_arr = tokenized_corpus[i]
    
    #Vocabulary
    for tokken in token_arr:
        if tokken not in uniq_words:
            uniq_words.append(tokken)
    
    token_arr = ['<sos>'] * 5 + token_arr + ['<eos>'] * 5
    tokenized_corpus[i] = token_arr

# print(tokenized_corpus[2])
# print(len(uniq_words))


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
    def __init__(self):
        super(NeuralLM,self).__init__()
        self.l1 = torch.nn.Linear(5*300, 300),
        self.a1 = torch.nn.Tanh(),
        self.l2 = torch.nn.Linear(300, len(uniq_words))

    def forward(self, inp):
        inp = self.l1(inp)
        inp = self.a1(inp)
        inp = self.l2(inp)
        return inp
    
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
            central_word.append(sentence[i + context_size])

    concatenated_contexts = np.array(concatenated_contexts)
    return concatenated_contexts, central_word

train_gram_inp, train_cen_inp = process_sentences(train_data,word2vec_model,N_Gram)
print(train_cen_inp[0])



