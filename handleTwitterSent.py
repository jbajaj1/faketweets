import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data


def load_tweets(filename):
    X = []
    y = []
    r = open(filename, 'r')
    for line in r:
        line = line.split()
        y.append(line[0])
        X.append(line[1:])
    return X, y

class Vocab:
    #variables weren't working for some reason
    #p_t = 0   # Used for padding short sentences
    #s_t = 1   # Start-of-sentence token
    #e_t = 2   # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0


    def add_word(self, word):
        word = word.lower()
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1


    def add_sentence(self, sentence):
        sentence_len = 0
        #print(sentence)
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def sentence_to_vec(self, sentence):
        vec = []
        for word in sentence:
            word = word.lower()
            vec.append(self.to_index(word))
        return vec


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.output = nn.Linear(self.hidden_size, self.vocab_size)


    def arrToVec(self, X):
        xMat = []
        for sentence in X:
            xMat.append(twitterVoc.sentence_to_vec(sentence))
        xMat = torch.stack(xMat)
        return xMat

    def forward(self, X):
        print(X)
        xMat = self.arrToVec(X)
        emb = self.embedding(xMat)
        emb = self.dropout(emb)

        hidden_states, final_state = self.rnn(emb, init_hidden_state)

        hidden_states = self.dropout(hidden_states)

        output_dist = self.output(hidden_states)

        return output_dist, hidden_states, final_state



twitterVoc = Vocab("twitter")

tweets = load_tweets("../twitter_sentiment/semeval_train.txt")
tokenizedTweets = []
for t in tweets[0]:
    twitterVoc.add_sentence(t)
    tokenizedTweets.append(twitterVoc.sentence_to_vec(t))




print(twitterVoc.to_word(4))
print(twitterVoc.to_index("this"))

print(twitterVoc.num_words)


ourLSTM = LSTM(twitterVoc.num_words, 64, 64)


