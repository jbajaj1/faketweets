import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import f1_score
import argparse
from torch.utils.data import TensorDataset, DataLoader


def load_tweets(filename, Voc, initVoc=False):
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
            Voc.add_sentence(t)
    for l in y:
        tokenizedLabels.append(labelDic[l])
    for t in X:
        tokenizedTweets.append(Voc.sentence_to_vec(t))

    tokenizedTweets = torch.LongTensor(tokenizedTweets)
    tokenizedLabels = torch.LongTensor(tokenizedLabels)

    return tokenizedTweets, tokenizedLabels


class Vocab:


    def __init__(self, name):
        self.name = name
        self.word2index = {"UNK":1, "ATUSER":2, "HTTPTKN":3}
        self.word2count = {"ATUSER" : 0, "HTTPTKN":0}
        self.index2word = {0: "PAD", 1: "UNK", 2: "ATUSER", 3: "HTTPTKN"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0
        self.unknown_count = 0


    def add_word(self, word):
        word = word.lower()
        if "@" == word[0] and len(word) != 1:
            self.word2count["ATUSER"] += 1
        elif word.startswith("http") and len(word) > 4:
            self.word2count["HTTPTKN"] += 1
        elif word not in self.word2index:
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
        #sentence = sentence.split("...").split(".").split("[").split("]").split("#").split("^^")
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
        if "@" == word[0] and len(word) != 1:
            return 2
        elif word.startswith("http") and len(word) > 4:
            return 3
        if word not in self.word2index:
            #print("Unknown word:", word)
            self.unknown_count += 1
            return 1
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
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2, dropout=0.1):
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
    neg_F1 = f1_score(expected == 0, predictions == 0, average="binary")
    pos_F1 = f1_score(expected == 2, predictions == 2, average="binary")
    F1 = f1_score(expected, predictions, average="weighted")
    return (expected == predictions).sum().item()/len(expected), confusion_matrix(expected, predictions), neg_F1, pos_F1, F1



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--emb_size', default=64, type=int)
    parser.add_argument('--hid_size', default=64, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--epochs', default=30, type=int)
    return parser.parse_args()


def main():
    args = parseargs()   
    '''
    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)
    '''
    twitterVoc = Vocab("twitter")

    #Put proper location of file here
    tokenizedTweets, tokenizedLabels = load_tweets(args.train, Voc=twitterVoc, initVoc=True)
    ourLSTM = LSTM(twitterVoc.num_words, args.emb_size, args.hid_size, args.num_layers, args.dropout)      

    print(twitterVoc.to_word(4))
    print(twitterVoc.to_index("this"))

    print(twitterVoc.num_words)

    opt = torch.optim.Adam(ourLSTM.parameters(), lr=.1)
    loss = torch.nn.CrossEntropyLoss()
    dataset = DataLoader(TensorDataset(tokenizedTweets, tokenizedLabels), batch_size=100)
    for i in range(args.epochs):
        print("Training on epoch", i)
        for batchidx, (x, y) in enumerate(dataset):
            opt.zero_grad()
            outputs = ourLSTM(x)
            lossVal = loss(outputs, y)
            lossVal.backward()
            opt.step()
                    
    #torch.save(ourLSTM.state_dict(), f'./models/main_run.model')

    tokTestTweets, tokTestLabels = load_tweets(args.test, twitterVoc)
    with torch.no_grad():
        predVal = ourLSTM(tokTestTweets).argmax(dim=-1)
        prec, conf, neg_F1, pos_F1, F1 = validate(tokTestLabels, predVal)
        print(f'Precision with emb_size[{args.emb_size}], hid_size[{args.hid_size}], layers[{args.num_layers}], and dropout[{args.dropout}]: {prec}\nF1: {F1}\nNegF1: {neg_F1}\nPosF1: {pos_F1}')





if __name__ == '__main__':
    main()
