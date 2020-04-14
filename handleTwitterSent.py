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
        X.append(line[0])
        y.append(line[1:])
    return X, y

class Vocab:

    p_t = 0   # Used for padding short sentences
    s_t = 1   # Start-of-sentence token
    e_t = 2   # End-of-sentence token

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

twitterVoc = Vocab("twitter")

tweets = load_tweets("./twitter_sentiment/semeval_train.txt")

for t in tweets[1]:
    twitterVoc.add_sentence(t)




print(twitterVoc.to_word(4))
print(twitterVoc.to_index("this"))

print(twitterVoc.num_words)
