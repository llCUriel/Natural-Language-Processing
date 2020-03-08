from pprint import pprint
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def readMessages(filename='../../Corpus/SMS_Spam_Corpus.txt',
                 encoding='latin-1'):
    smss = []
    tags = []
    with open(filename, encoding=encoding) as file:
        for line in file:
            if len(line) > 2:
                i = line.rindex(',')
                sms, tag = line[:i-1], line[i+1:]
                if tag.strip().lower() == 'spam':
                    tag = 0
                else:
                    tag = 1
                tags.append(tag)
                smss.append(sms)

        return smss, tags


# clf = MultinomialNB()
# clf.fit(X, y)

# pprint(X)
# print(clf.predict(X[2:3]))

if __name__ == "__main__":
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    smss, tags = readMessages()
    countVector = CountVectorizer()
    pprint(smss)
    smsCounts = countVector.fit_transform(smss)
    model = MultinomialNB()
    pprint(smsCounts[:100])
    model.fit(smsCounts, tags)

    print(model.predict(smss[101]))
    print(tags[101])
