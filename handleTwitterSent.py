import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from keras.utils import Progbar

import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import f1_score
import argparse
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def load_tweets(filename, Voc, initVoc=False):
    X = []
    y = []
    r = open(filename, "r")
    for line in r:
        line = line.split()
        y.append(line[0])
        X.append(line[1:])

    tokenizedTweets = []
    tokenizedLabels = []
    labelDic = {"negative": 0, "neutral": 1, "positive": 2}
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
        self.word2index = {"UNK": 1, "ATUSER": 2, "HTTPTKN": 3}
        self.word2count = {"ATUSER": 0, "HTTPTKN": 0}
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
        # print(sentence)
        # sentence = sentence.split("...").split(".").split("[").split("]").split("#").split("^^")
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
            # print("Unknown word:", word)
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
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            self.embedding_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.output = nn.Linear(self.hidden_size, 3)

    def forward(self, X):
        emb = self.embedding(X)
        emb = self.dropout(emb)

        hidden_states, _ = self.lstm(emb)

        numMask = (X != 0).float()
        mask = (X != 0).float().unsqueeze(-1).expand(hidden_states.size())
        hidden_states = (hidden_states * mask).sum(-2) / numMask.sum(-1).unsqueeze(-1)
        hidden_states = self.dropout(hidden_states)

        output_dist = self.output(hidden_states)

        return output_dist


class StackedConvLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_sizes=[]):
        super().__init__()
        self.layers = nn.ModuleList()
        for k in kernel_sizes:
            assert k % 2 == 1
            self.layers.append(nn.Conv1d(in_features, out_features, k, padding=k // 2))

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], 1)


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size=64, hidden_sizes=[64], kernel_sizes=[1,3,5], dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

        conv_layers = []
        prev_size = self.embedding_size
        for h in hidden_sizes:
            conv_layers.append(
                StackedConvLayer(
                    prev_size, h, kernel_sizes=kernel_sizes,
                )
            )
            prev_size = len(kernel_sizes) * h

        self.conv_layer = nn.Sequential(*conv_layers)

        self.output = nn.Linear(prev_size, 3)

    def forward(self, X):
        emb = self.embedding(X)
        emb = self.dropout(emb)
        emb = emb.permute(0, 2, 1)

        hidden_states = F.relu(self.conv_layer(emb))

        hidden_states = hidden_states.permute(0, 2, 1)


        numMask = (X != 0).float()
        mask = (X != 0).float().unsqueeze(-1).expand(hidden_states.size())
        hidden_states = (hidden_states * mask).sum(-2) / numMask.sum(-1).unsqueeze(-1)
        hidden_states = self.dropout(hidden_states)

        output_dist = self.output(hidden_states)

        return output_dist


def validate(expected, predictions):
    neg_F1 = f1_score(expected == 0, predictions == 0, average="binary")
    pos_F1 = f1_score(expected == 2, predictions == 2, average="binary")
    F1 = f1_score(expected, predictions, average="weighted")
    return (
        (expected == predictions).sum().item() / len(expected),
        confusion_matrix(expected, predictions),
        neg_F1,
        pos_F1,
        F1,
    )


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--emb-size", default=64, type=int)
    parser.add_argument("--hid-size", default=64, type=int, help="use for lstm")
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--kernel-sizes", nargs="+", type=int, default=[1,3,5], help="kernel sizes for stacked conv layer")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64],  help="hidden sizes for multi conv layer (use for cnn)")
    parser.add_argument("--classifier", default="LSTM", type=str)
    return parser.parse_args()


def main():
    args = parseargs()
    """
    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)
    """
    twitterVoc = Vocab("twitter")

    # Put proper location of file here
    tokenizedTweets, tokenizedLabels = load_tweets(args.train, Voc=twitterVoc, initVoc=True)
    if args.classifier == "LSTM":
        classifier = LSTM(
            twitterVoc.num_words, args.emb_size, args.hid_size, args.num_layers, args.dropout
        )
    elif args.classifier == "CNN":
        classifier = CNN(twitterVoc.num_words, args.emb_size, args.hidden_sizes, args.kernel_sizes, args.dropout)

    print(twitterVoc.to_word(4))
    print(twitterVoc.to_index("this"))

    print(twitterVoc.num_words)

    opt = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    dataset = DataLoader(TensorDataset(tokenizedTweets, tokenizedLabels), batch_size=args.batch_size)
    for i in range(args.epochs):
        p = Progbar(len(dataset))
        for batchidx, (x, y) in enumerate(dataset):
            opt.zero_grad()
            outputs = classifier(x)
            lossVal = loss_fn(outputs, y)
            stateful_metrics = []
            stateful_metrics.append((f"Train Loss on epoch {i}", lossVal.item()))
            lossVal.backward()
            opt.step()
            p.add(1, stateful_metrics)


    # torch.save(ourLSTM.state_dict(), f'./models/main_run.model')

    tokTestTweets, tokTestLabels = load_tweets(args.test, twitterVoc)
    with torch.no_grad():
        predVal = classifier(tokTestTweets).argmax(dim=-1)
        prec, conf, neg_F1, pos_F1, F1 = validate(tokTestLabels, predVal)
        if args.classifier == "LSTM":
            print(
                f"Precision with classifier[{args.classifier}], epochs[{args.epochs}], emb_size[{args.emb_size}], hid_size[{args.hid_size}], layers[{args.num_layers}], dropout[{args.dropout}], batch_size[{args.batch_size}], and learning rate[{args.lr}]: {prec}\nF1: {F1}\nNegF1: {neg_F1}\nPosF1: {pos_F1}"
            )
        elif args.classifier == "CNN":
            print(
                f"Precision with classifier[{args.classifier}], epochs[{args.epochs}], emb_size[{args.emb_size}], hidden_size(s)[{args.hidden_sizes}], kernel_size(s)[{args.kernel_sizes}], dropout[{args.dropout}, batch_size[{args.batch_size}, and learning rate[{args.lr}]: {prec}\nF1: {F1}\nNegF1: {neg_F1}\nPosF1: {pos_F1}"
            ) 


if __name__ == "__main__":
    main()
