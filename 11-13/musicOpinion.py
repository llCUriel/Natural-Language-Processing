import re
import os
# import resource
import string

import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from pprint import pprint
import mord


lemma = {}


def lemmanize(tokens, taggin=False):
    global lemma
    if len(lemma) == 0:
        # print("lemma init")
        with open('../generate.txt') as f:
            for l in f:
                s = l.replace('#', '')
                words = s.split()
                if len(words) > 2:
                    if taggin:
                        lemma[words[0]] = words[-1], str(words[-2]).lower()
                    else:
                        lemma[words[0]] = words[-1]

    lemmanized = []
    unknown = []
    for w in tokens:
        try:
            lemmanized.append(lemma[w])
        except KeyError:
            lemmanized.append(w)
            unknown.append(w)
    return lemmanized, unknown


def removeSpecialCharacters(tokens):
    '''Remove special characters, we receive tokens'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token)
                                    for token in tokens])
    return list(filtered_tokens)


def cleanText(text, lemmanized, cleaningLevel=1):
    ''' A string is cleaned, depending on the level of cleaning:
            0 -> raw string
            1 - > lower case, special characters removed
            2 - > Stopwords are removed
        the cleaned string is return
    '''
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('spanish')

    if cleaningLevel == 0:
        cleanedTokens = tokens
    elif cleaningLevel == 1:
        cleanedTokens = removeSpecialCharacters([
            t.lower() for t in tokens
            if t.isalpha() and t.lower() not in stopwords
        ])
    elif cleaningLevel == 2:  # Without stopwords
        cleanedTokens = removeSpecialCharacters([
            t.lower() for t in tokens if t.isalpha()
        ])
    if lemmanized:
        lemmanizedTokens, unkown = lemmanize(cleanedTokens)
        cleanedTokens = lemmanizedTokens
    cleanedText = ' '.join(cleanedTokens)

    return cleanedText


def readMessages(cleaningLevel=1, lemmanized=False,
                 encoding='latin-1'):
    opinions = []
    tags = []
    path = "musica/"
    file = open('x.txt')
    for line in file:
        opinion = cleanText(line.strip(), lemmanized, cleaningLevel)
        opinions.append(opinion)
    file.close()
    file = open('y.txt')
    for line in file:
        tag = int(line.strip())
        tags.append(tag)
    file.close()

    return opinions, tags


def testModel(X, y, model, size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    clf = model

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy of prediction is', clf.score(X_test, y_test))
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    targetNames = ['yes', 'no']
    report = metrics.classification_report(
        y_test, y_pred,
        target_names=targetNames,
        output_dict=True
    )
    result = {
        targetNames[0]: report[targetNames[0]],
        targetNames[1]: report[targetNames[1]],
    }
    return result


if __name__ == '__main__':
    models = [
        mord.LogisticIT()
    ]
    cleaningLevels = [2]
    cleaningDescription = [
        "Tokens originales",
        "Tokens con letras",
        "Tokens con letras sin stopwords"
    ]

    for model in models:
        for cl in cleaningLevels:
            for lemmanized in [True]:
                # list of texts, each text is a string (a sms)
                sampleTexts, y = readMessages(cl, lemmanized)

                # print(len(sampleTexts), "messages in corpus")
                # print(y.count(0), " spam messages in corpus")
                # print(y.count(1), " ham messages in corpus")

                # Build vector of token counts
                count_vect = CountVectorizer()
                X = count_vect.fit_transform(sampleTexts)

                # print("Testing  model ", type(model))
                # print(cleaningDescription[cl])
                # print("Lemmatized ", lemmanized)
                result = testModel(X, y, model)
                # print('class   f1-score     precision   recall    support')
                # print("spam ", result['spam']['f1-score'],
                      # result['spam']['precision'], result['spam']['recall'],
                      # result['spam']['support'])
                # print("ham  ", result['ham']['f1-score'],
                      # result['ham']['precision'], result['ham']['recall'],
                      # result['ham']['support'])
                # pprint(result)
                # print("--"*30)
                # print()
                # print()
