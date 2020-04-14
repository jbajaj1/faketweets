import csv

class TweetsClassifier:


    def convert_to_matrix(self, trainX):
        F = []
        for x in trainX:
            features = [
                preposition,
                noun,
                the,
                len(x[3]),
                len(x[5]),
                int(x[8]),
                int(x[9]),
                int(x[10])
            ]
            F.append(np.array(features))

        return np.array(F)

    def vectify_labels(self, trainY):
        yArr = []
        for y in trainY:
            if y == "Fake":
                yArr.append(1)
            else:
                yArr.append(0)
        return np.array(yArr)

    def train(self, trainX, trainY):
        F = self.convert_to_matrix(trainX)
        vecs = self.vectify_labels(trainY)
        self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 64), random_state=1, verbose=True, max_fun=50000, max_iter=500)
        self.clf.fit(F, vecs)

    def classify(self, testX):
        F = self.convert_to_matrix(testX)
        yClasses = self.clf.predict(F)
        yArr = []
        for y in yClasses:
            if y == 1:
                yArr.append("Fake")
            else:
                yArr.append("Real")
        return yArr


#Needs updating
class UserClassifier:
    def convert_to_matrix(self, trainX):
        F = []
        for x in trainX:

            abbrev = []
            timeterm = []
            title = []
            sentenceinternal = []
            unlikely_prop_noun = []
            alphaArr = []
            for i in range(1, 8):
                x[i] = x[i].lower()
                if i == 4:
                    continue
                if x[i] in self.abbrevs:
                    abbrev.append(1)
                else:
                    abbrev.append(0)
                if x[i] in self.timeterms:
                    timeterm.append(1)
                else:
                    timeterm.append(0)
                if x[i] in self.titles:
                    title.append(1)
                else:
                    timeterm.append(0)
                if x[i] in self.sentence_internals:
                    sentenceinternal.append(1)
                else:
                    sentenceinternal.append(0)
                if x[i] in self.unlikely_prop_nouns:
                    unlikely_prop_noun.append(1)
                else:
                    unlikely_prop_noun.append(0)
                if x[i].isalpha():
                    alphaArr.append(1)
                else:
                    alphaArr.append(0)
            features = [
                preposition,
                noun,
                the,
                len(x[3]),
                len(x[5]),
                isUpper,
                *abbrev,
                *timeterm,
                *title,
                *sentenceinternal,
                *unlikely_prop_noun,
                *alphaArr,
                int(x[8]),
                int(x[9]),
                int(x[10])
            ]
            F.append(np.array(features))

        return np.array(F)

    def vectify_labels(self, trainY):
        yArr = []
        for y in trainY:
            if y == "Fake":
                yArr.append(1)
            else:
                yArr.append(0)
        return np.array(yArr)

    def train(self, trainX, trainY):
        F = self.convert_to_matrix(trainX)
        vecs = self.vectify_labels(trainY)
        self.clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 64), random_state=1, verbose=True, max_fun=50000, max_iter=500)
        self.clf.fit(F, vecs)

    def classify(self, testX):
        F = self.convert_to_matrix(testX)
        yClasses = self.clf.predict(F)
        yArr = []
        for y in yClasses:
            if y == 1:
                yArr.append("Fake")
            else:
                yArr.append("Real")
        return yArr

def load_tweets(filename):
    with open(filename, newline="") as f:
        X = []
        y = []
        r = csv.reader(f, delimiter=",")
        first = True
        for row in r:
            if first:
                first = False
            else:
                X.append(row[0])
                y.append(row[1:])

        return X, y




#Work in Progress
def transform_538(filename):
    with open(filename, newline="") as f:
        X = []
        y = []
        r = csv.reader(f, delimiter=",")
        first = True
        for row in r:
            if first:
                first = False
            else:
                X.append(row[16])
                y.append(row[6:15])

        return X, y


print(load_tweets("IRAhandle_tweets_7.csv")[0])