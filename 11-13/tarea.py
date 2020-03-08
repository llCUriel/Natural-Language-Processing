# -*- coding: utf-8 -*-
import re
# import resource
import string
from clean_tokens import get_text_string
import nltk
# read path
from os import scandir, getcwd
from os.path import abspath

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
import numpy as np

lemma = {}

def lemmanize(tokens, taggin=False):
    global lemma
    if len(lemma) == 0:
        print("lemma init")
        with open('/home/euron/Downloads/generate.txt', encoding='latin1') as f:
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

def ls(ruta = '/home/euron/Desktop/music'):
    #return [arch.name for arch in scandir(ruta) if arch.is_file()
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]

def readMessages(cleaningLevel=1, lemmanized=False,
                 filename=ls(),
                 encoding='latin1',):
    smss = []
    tags = []

    for f in filename:
        sms = get_text_string(f)
       # print(f[26])
        
        if f[26] == 'n':
            tag = 0
        else:
            tag = 1
        sms = cleanText(sms, lemmanized, cleaningLevel)    
        smss.append(sms)
        tags.append(tag)
    return smss, tags


def testModel(X, y, model, size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)

  #  print(X, y)
    print("YTRAIN:"+str(y_test))
    
    clf = model

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print('Accuracy of prediction is', clf.score(X_test, y_test))
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    targetNames = ['No', 'Yes']
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
        #MultinomialNB(),
        mord.LogisticIT()
    ]
    cleaningLevels = [2]
    cleaningDescription = [
        "Tokens originales"
    ]

    for model in models:
        for cl in cleaningLevels:
            for lemmanized in [False]:
                # list of texts, each text is a string (a sms)
                sampleTexts, y = readMessages(cl, lemmanized)
                count_vect = CountVectorizer()
                X = count_vect.fit_transform(sampleTexts) 
              #  print(y)
                y=np.asarray(y)
               # print(X)
              #  print("REALYTRAIN"+str(y))
                # print(len(sampleTexts), "messages in corpus")
                # print(y.count(0), " spam messages in corpus")
                # print(y.count(1), " ham messages in corpus")

                # Build vector of token counts
               
                #Y= count_vect.fit_transform(y)
                print("Testing  model ", type(model))
                #sprint(cleaningDescription[cl])
                #print("Lemmatized ", lemmanized)
                #print(y)
                result = testModel(X, y, model)
                print('class   f1-score     precision   recall    support')
                print("No ", result['No']['f1-score'],
                      result['No']['precision'], result['No']['recall'],
                      result['No']['support'])
                print("Yes  ", result['Yes']['f1-score'],
                      result['Yes']['precision'], result['Yes']['recall'],
                      result['Yes']['support'])
                # pprint(result)
                print("--"*30)
                print()
                print()
                