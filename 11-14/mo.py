import re
import os
# import resource
import string

import nltk

import untangle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from pprint import pprint
import numpy as np
import mord

# lemma = {}

def lemmanize(tokens, lemmas='../generate.txt', taggin=False):
    # global lemma
    lemma = {}
    if len(lemma) == 0:
        # print("lemma init")
        with open(lemmas, encoding='latin-1') as f:
            for l in f:
                words = l.strip().split()
                if len(words) > 2:
                    if words[-1] == '1':
                        if taggin:
                            lemma[words[0]] = str(words[1]).lower()
                        else:
                            lemma[words[0]] = words[0]

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


def cleanText(text, lemmanized, cleaningLevel=1, lemmas='../generate.txt'):
    ''' A string is cleaned, depending on the level of cleaning:
            0 -> raw string
            1 - > lower case, special characters removed
            2 - > Stopwords are removed
        the cleaned string is return
    '''
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('spanish')

    if cleaningLevel == 0:
        cleanedTokens = tokens.split()
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
        lemmanizedTokens, unkown = lemmanize(cleanedTokens, lemmas)
        cleanedTokens = lemmanizedTokens
    cleanedText = ' '.join(cleanedTokens)

    return cleanedText


def readMessages(cleaningLevel=1, lemmanized=False,
                 encoding='latin-1'):
    opinions = []
    tags = []
    path = "./corpusCriticasCine/"

    for filename in os.listdir(path):
        if filename.endswith('xml'):
            lemmas = path+filename.split('.')[0]+'.review.pos'
            obj = untangle.parse(lemmas)
            tag = int(obj.review['rank'])
            opinion = cleanText(obj.review.body, lemmanized, cleaningLevel,
                                lemmas)

            opinions.append(opinion)
            tags.append(tag)

    return opinions, tags


def testModel(X, y, model, size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

    clf = model

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy of prediction is', clf.score(X_test, y_test))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    targetNames = ['1', '2', '3', '4', '5']
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
        mord.LogisticIT(alpha=1.0)
    ]
    cleaningLevels = [0]
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

                print(len(sampleTexts), "messages in corpus")
                print(y.count(1), " 1 messages in corpus")
                print(y.count(2), " 2 messages in corpus")
                print(y.count(3), " 3 messages in corpus")
                print(y.count(4), " 4 messages in corpus")
                print(y.count(5), " 5 messages in corpus")

                # Build vector of token counts
                count_vect = CountVectorizer()
                X = count_vect.fit_transform(sampleTexts)
                y = np.asarray(y)
                print("Testing  model ", type(model))
                print(cleaningDescription[cl])
                print("Lemmatized ", lemmanized)
                result = testModel(X, y, model)
                print('class   f1-score     precision   recall    support')
                print("1 ", result['1']['f1-score'],
                      result['1']['precision'], result['1']['recall'],
                      result['1']['support'])
                print("2  ", result['2']['f1-score'],
                      result['2']['precision'], result['2']['recall'],
                      result['2']['support'])
                print("3  ", result['3']['f1-score'],
                      result['3']['precision'], result['3']['recall'],
                      result['3']['support'])
                print("4  ", result['4']['f1-score'],
                      result['4']['precision'], result['4']['recall'],
                      result['4']['support'])
                print("5  ", result['5']['f1-score'],
                      result['5']['precision'], result['5']['recall'],
                      result['5']['support'])
                pprint(result)
                print("--"*30)
                print()
                print()
