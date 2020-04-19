import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data

from torch.utils.data import TensorDataset, DataLoader


def load_tweets(filename, initVoc=False):
    X = []
    y = []
    r = open(filename, 'r')
    for line in r:
        line = line.split()
        y.append(line[0])
        X.append(line[1:])

    tokenizedTweets = []
    tokenizedLabels = []
    labelDic = {"negative":0, "neutral":1, "positive":2}
    if initVoc:
        for t in X:
            twitterVoc.add_sentence(t)
    for l in y:
        tokenizedLabels.append(labelDic[l])
    for t in X:
        tokenizedTweets.append(twitterVoc.sentence_to_vec(t))

    tokenizedTweets = torch.LongTensor(tokenizedTweets)
    tokenizedLabels = torch.LongTensor(tokenizedLabels)

    return tokenizedTweets, tokenizedLabels


class Vocab:


    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "UNK"}
        self.num_words = 2
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
        if word not in self.word2index:
            return self.word2index["UNK"]
        return self.word2index[word]

    def sentence_to_vec(self, sentence):
        vec = []
        for word in sentence:
            word = word.lower()
            vec.append(self.to_index(word))
        while len(vec) < self.longest_sentence:
            vec.append(0)
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
        self.output = nn.Linear(self.hidden_size, 3)



    def forward(self, X):
        emb = self.embedding(X)
        emb = self.dropout(emb)

        hidden_states, _ = self.rnn(emb)

        numMask = (X != 0).float()
        mask = (X != 0).float().unsqueeze(-1).expand(hidden_states.size())
        hidden_states = (hidden_states*mask).sum(-2)/numMask.sum(-1).unsqueeze(-1)
        hidden_states = self.dropout(hidden_states)

        output_dist = self.output(hidden_states)

        return output_dist

def validate(expected, predictions):
    '''
    totdiff = 0
    numdiff = 0
    counter = 0
    for i in expected:
        diff = abs(i-predictions[counter])
        totdiff += diff
        if diff != 0:
            numdiff += 1
        counter += 1
    return totdiff, numdiff
    '''
    return confusion_matrix(expected, predictions)






twitterVoc = Vocab("twitter")

tokenizedTweets, tokenizedLabels = load_tweets("../twitter_sentiment/semeval_train.txt",  initVoc=True)

print(twitterVoc.to_word(4))
print(twitterVoc.to_index("this"))

print(twitterVoc.num_words)

ourLSTM = LSTM(twitterVoc.num_words, 64, 64)

#ourLSTM.forward(tokenizedTweets)


##################
#####Training#####
##################

opt = torch.optim.Adam(ourLSTM.parameters(), lr=.1)
loss = torch.nn.CrossEntropyLoss()
epochs = 100
dataset = DataLoader(TensorDataset(tokenizedTweets, tokenizedLabels), batch_size=100)
for i in range(epochs):
    print("Training on epoch", i)
    for batchidx, (x, y) in enumerate(dataset):
        opt.zero_grad()
        outputs = ourLSTM(x)
        lossVal = loss(outputs, y)
        lossVal.backward()
        opt.step()


##################
#####Predict######
##################


predVal = ourLSTM(tokenizedTweets).argmax(dim=-1)

print("Results for Train Data:", validate(tokenizedLabels, predVal))


filelist = ["../twitter_sentiment/Twitter2013_raw.txt", "../twitter_sentiment/Twitter2013_raw", "../twitter_sentiment/Twitter2015_raw", "../twitter_sentiment/Twitter2016_raw"]

for file in filelist:
    tokTestTweets, tokTestLabels = load_tweets(file)
    predVal = ourLSTM(tokTestTweets).argmax(dim=-1)
    print("Results for", file, validate(tokTestLabels, predVal))







