#<-----------------------------------------Importing Libraries------------------------------------------------->

import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
# nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

#<-----------------------------------------Importing and Cleaning Dataset---------------------------------------------------->

with open('./datasets/Auguste_Maquet.txt', 'r', encoding='utf-8') as file:
    corpus = file.readlines()

# Clean the data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()  # Convert to lowercase
    return text

cleaned_corpus = [clean_text(sentence) for sentence in corpus]
# print(len(cleaned_corpus))
# print(cleaned_corpus[1])

#<---------------------------------------------Tokenization and Emmbedding----------------------------------------------------->

tokenized_corpus = [word_tokenize(sentence) for sentence in cleaned_corpus]
# print(tokenized_corpus[1])

word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=200, window=5, min_count=1, workers=4)

# similarity = word2vec_model.wv.similarity('king', 'ebook')
# print(similarity)





