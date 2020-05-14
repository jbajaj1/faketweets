import argparse
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import f1_score, confusion_matrix


# negative : 0
# neutral : 1
# positive : 2


class Classifier:
    def __init__(self):
        self.V1 = None
        self.V2 = None
        self.V3 = None
        self.numwords = 1
        self.word2index = {"unk": 0}
        self.index2word = {0: "unk"}

    def add_word(self, word, train, x_dic):
        if train:
            if word not in self.word2index:
                self.word2index[word] = self.numwords
                self.index2word[self.numwords] = word
                self.numwords += 1
            if self.word2index[word] not in x_dic:
                x_dic[self.word2index[word]] = 1
            else:
                x_dic[self.word2index[word]] += 1
        else:
            if word in self.word2index:
                if self.word2index[word] not in x_dic:
                    x_dic[self.word2index[word]] = 1
                else:
                    x_dic[self.word2index[word]] += 1
            else:
                if self.word2index["unk"] not in x_dic:
                    x_dic[self.word2index["unk"]] = 1
                else:
                    x_dic[self.word2index["unk"]] += 1

    def make_vocab(self, trainX, train=True):
        trainXFreqs = []
        for x in trainX:
            x_dic = {}

            for word in x:
                self.add_word(word, train, x_dic)

            trainXFreqs.append(x_dic)

        return trainXFreqs

    def make_centroids(self, trainX, trainY):
        V1_arr = []
        V2_arr = []
        V3_arr = []
        for idx, y in enumerate(trainY):
            if y == 0:
                V1_arr.append(trainX[idx])
            elif y == 1:
                V2_arr.append(trainX[idx])
            elif y == 2:
                V3_arr.append(trainX[idx])
            else:
                raise ValueError("y should only be 1, 2, or 3")

        C1 = {}
        C2 = {}
        C3 = {}
        for ind in self.index2word:
            for v in V1_arr:
                if ind in v:
                    if ind not in C1:
                        C1[ind] = v[ind]
                    else:
                        C1[ind] += v[ind]
            for v in V2_arr:
                if ind in v:
                    if ind not in C2:
                        C2[ind] = v[ind]
                    else:
                        C2[ind] += v[ind]
            for v in V3_arr:
                if ind in v:
                    if ind not in C3:
                        C3[ind] = v[ind]
                    else:
                        C3[ind] += v[ind]

        C1_arr = []
        C2_arr = []
        C3_arr = []

        numS1 = len(V1_arr)
        numS2 = len(V2_arr)
        numS3 = len(V3_arr)

        for i in range(self.numwords):
            if i in C1:
                C1_arr.append(C1[i] / numS1)
            else:
                C1_arr.append(0)
            if i in C2:
                C2_arr.append(C2[i] / numS2)
            else:
                C2_arr.append(0)
            if i in C3:
                C3_arr.append(C3[i] / numS3)
            else:
                C3_arr.append(0)

        self.V1 = np.array(C1_arr)
        self.V2 = np.array(C2_arr)
        self.V3 = np.array(C3_arr)

    def train(self, trainX, trainY):
        trainXFreqs = self.make_vocab(trainX)
        self.make_centroids(trainXFreqs, trainY)

    def make_x_vec(self, x):
        x_vec = []
        for i in range(self.numwords):
            if i in x:
                x_vec.append(x[i])
            else:
                x_vec.append(0)
        return np.array(x_vec)

    def classify(self, testX):
        testXFreqs = self.make_vocab(testX, train=False)

        yArr = []

        for x in testXFreqs:
            sim1 = 1 - distance.cosine(self.V1, self.make_x_vec(x))
            sim2 = 1 - distance.cosine(self.V2, self.make_x_vec(x))
            sim3 = 1 - distance.cosine(self.V3, self.make_x_vec(x))

            if sim1 > sim2 and sim1 > sim3:
                yArr.append(0)
            elif sim2 > sim1 and sim2 > sim3:
                yArr.append(1)
            else:
                yArr.append(2)

        return np.array(yArr)


def tokenize_y(y):
    y_tok = []
    for label in y:
        if label == "negative":
            y_tok.append(0)
        elif label == "neutral":
            y_tok.append(1)
        elif label == "positive":
            y_tok.append(2)
        else:
            raise ValueError("y must be 0 1 or 2")
    return np.array(y_tok)


def load_data(file):
    X = []
    y = []
    r = open(file, "r")

    for line in r:
        line = line.split()
        y.append(line[0])
        X.append(line[1:])

    y = tokenize_y(y)
    return X, y


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
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    classifier = Classifier()

    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, "w") as fout:
            for output in outputs:
                print(output, file=fout)

    prec, conf, neg_F1, pos_F1, F1 = validate(outputs, testY)

    print(f"Accuracy: {prec}\nF1: {F1}\nneg_F1: {neg_F1}\npos_F1: {pos_F1}")


if __name__ == "__main__":
    main()
