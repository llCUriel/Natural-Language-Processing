import re
import os
# import resource
import string

import pickle
import nltk

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from pprint import pprint
import numpy as np
import mord


def savePkl(tagger, filename="name.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(tagger, f, -1)


def loadPkl(filename="name.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)



def lemmanize(tokens, lemmas='../generate.txt', polarity=True):
    lemma = {}
    if len(lemma) == 0:
        with open(lemmas, encoding='latin-1') as f:
            for l in f:
                words = l.strip().split()
                if len(words) > 2:
                    if words[-1] == '1':
                        if polarity:
                            if words[-1] == '-':
                                pol = int(words[-2])
                            else:
                                pol = int(words[-1])
                            lemma[words[0]] = (str(words[1]).lower(), pol)
                        else:
                            lemma[words[0]] = words[0]

    lemmanized = []
    unknown = []
    pol = 0
    for w in tokens:
        try:
            lemmanized.append(lemma[w][0])
            if polarity:
                pol += lemma[w][1]
        except KeyError:
            lemmanized.append(w)
            unknown.append(w)
    return lemmanized, unknown, pol


polarities = {}


def polarize(tokens):
    global polarities
    if len(polarities) == 0:
        # Check if the file tagger.pkl exists
        # if so load tagger, if not create a tagger
        if os.path.isfile('polarities.pkl'):
            polarities = loadPkl('polarities.pkl')
            print("polarities loaded")
        else:
            print("polarities init")
            dicts = ['../fullStrengthLexicon.txt',
                     '../mediumStrengthLexicon.txt']
            for d in dicts:
                with open(d, encoding='latin-1') as f:
                    for line in f:
                        words = line.strip().split()
                        if len(words) > 2:
                            if words[-1] == 'pos':
                                polarities[words[0]] = 1
                            elif words[-1] == 'neg':
                                polarities[words[0]] = -1
            savePkl(polarities, 'polarities.pkl')

    polarized = []
    for w in tokens:
        try:
            polarized.append(polarities[w])
        except KeyError:
            # If the word is unknown the polarity is neutral
            polarized.append(0)
    return polarized


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
        lemmanizedTokens, unkown, polarity = lemmanize(cleanedTokens, lemmas)
        cleanedTokens = lemmanizedTokens
    cleanedText = ' '.join(cleanedTokens)

    return cleanedText, polarity


def readMessages(cleaningLevel=1, lemmanized=False,
                 encoding='latin-1'):
    global parser
    opinions = []
    tags = []
    path = "../corpusCriticasCine/"
    stats = [[0, 0, 0] for i in range(5)]

    if os.path.isfile('opinions.pkl') and os.path.isfile('tags.pkl'):
        opinions = loadPkl('opinions.pkl')
        print("opinions loaded")
        tags = loadPkl('tags.pkl')
        print("tags loaded")
        stats = loadPkl('stats.pkl')
        print("stats loaded")
    else:
        print("messages init")
        for filename in sorted(os.listdir(path)):
            if filename.endswith('xml'):
                lemmas = path+filename.split('.')[0]+'.review.pos'
                # print(filename, lemmas)
                with open(path+filename, encoding=encoding) as f:
                    soup = BeautifulSoup(f.read(), 'xml')
                    tag = int(soup.find('review').attrs['rank'])
                    opinion = soup.find('body').getText()
                    opinion, polarity = cleanText(opinion, lemmanized,
                                                  cleaningLevel, lemmas)
                    opinions.append(opinion)
                    tags.append(tag)
                polarized = polarize(opinion.split())
                # print("P: ", polarized.count(1), "N: ", polarized.count(-1))
                stats[tag-1][0] += polarity
                stats[tag-1][1] += polarized.count(1)
                stats[tag-1][2] += polarized.count(-1)
        savePkl(opinions, 'opinions.pkl')
        savePkl(tags, 'tags.pkl')
        savePkl(stats, 'stats.pkl')

    return opinions, tags, stats


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
        targetNames[2]: report[targetNames[2]],
        targetNames[3]: report[targetNames[3]],
        targetNames[4]: report[targetNames[4]],
    }
    return result


if __name__ == '__main__':
    model = mord.LogisticIT(alpha=1.0)
    cl = 0
    lemmanized = True
    cleaningDescription = [
        "Tokens originales",
        "Tokens con letras",
        "Tokens con letras sin stopwords"
    ]

    # list of texts, each text is a string (a sms)
    sampleTexts, y, stats = readMessages(cl, lemmanized)

    print(len(sampleTexts), "messages in corpus")
    print(y.count(1), " 1 messages in corpus")
    print(y.count(2), " 2 messages in corpus")
    print(y.count(3), " 3 messages in corpus")
    print(y.count(4), " 4 messages in corpus")
    print(y.count(5), " 5 messages in corpus")

    for i in range(len(stats)):
        size = y.count(i+1)
        # print(size, stats[i][1], stats[i][2])
        print('-'*20)
        print("Categoria ", i)
        print('Has ', size, ' reviews')
        print('Pos:', stats[i][1]/size)
        print('Neg:', stats[i][2]/size)

    '''
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
    '''
