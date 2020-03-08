import re
import os
# import resource
import string
from collections import Counter

import pickle
import nltk
import operator

from bs4 import BeautifulSoup
from pprint import pprint
import numpy as np


def savePkl(tagger, filename="name.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(tagger, f, -1)


def loadPkl(filename="name.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def lemmanize(tokens, taggin=False):
    global lemma
    global tagger

    if len(lemma) == 0:
        # print("lemma init")
        with open('../generate.txt', encoding='latin-1') as f:
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
            if taggin:
                lemmanized.append((w, 'nounArtificial'))
                unknown.append((w, 'nounArtificial'))
            else:
                lemmanized.append(w)
                unknown.append(w)
    return lemmanized, unknown

def delete_diacritic_marks(word):
    import re
    word = word.lower()
    word = re.sub('[á]', 'a', word)
    word = re.sub('[é]', 'e', word)
    word = re.sub('[í]', 'i', word)
    word = re.sub('[ó]', 'o', word)
    word = re.sub('[ú]', 'u', word)
    word = re.sub('[ñ]', 'n', word)
    return word


def removeSpecialCharacters(tokens):
    '''Remove special characters, we receive tokens'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filteredTokens = filter(None, [pattern.sub(' ', token)
                                   for token in tokens])

    return list(filteredTokens)


def cleanText(text, lemmanized=False, cleaningLevel=1, taggin=False):
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
            t.lower() for t in tokens if t.isalpha()
        ])
    elif cleaningLevel == 2:  # Without stopwords
        cleanedTokens = removeSpecialCharacters([
            t.lower() for t in tokens
            if t.isalpha() and t.lower() not in stopwords
        ])
    if taggin:
        sentences = nltk.sent_tokenize(text, language='spanish')
        tagged = []
        tokens = []
        # Tags using the tagger, as a fallback,
        # it uses the tag from generator.txt dictonary
        for sentence in sentences:
            taggedTokens = tagger.tag(nltk.word_tokenize(sentence, 'spanish'))
            for token, tag in taggedTokens:
                token = str(token).lower()
                tag = str(tag).lower()
                # When the token was a punctuation character token is empty
                if token.isalpha() and token not in stopwords:
                    lemmanized, unknown = lemmanize([token], taggin)
                    # If it's a token that isn't on the dictonary
                    if len(unknown) == 1:
                        lemmanized[0] = lemmanized[0], 'nounArtificial'
                    else:
                        tag = lemmanized[0][1]
                    tokens.append(token)
                    tagged.append((token, tag))
        return tokens, tagged
    else:
        lemmanizedTokens, unkown = lemmanize(cleanedTokens, taggin)
        cleanedText = ' '.join(cleanedTokens)
        return cleanedText


def readTagMessages(path, tagger=None, cleaningLevel=1, lemmanized=False,
                    encoding='latin-1'):
    reviews = []
    tagged = []
    classifications = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith('txt'):
            with open(path+filename, encoding=encoding) as f:
                review = f.read().strip()
                x = review
                corpus = cleanText(review, False, 0, False)
                review, tags = cleanText(review, lemmanized,
                                         cleaningLevel, True)
                # review = get_top_ngrams(review)
                classification = filename.split('.')[0].split('_')
                tagged.append(tags)
                reviews.append(x)
                classifications.append((filename, classification))

    return tagged, classifications, reviews, corpus


def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] for index in range(n)])

def get_top_ngrams(tokens, ngram_val=1, limit=5):
    # cleanText(corpus)
    # tokens = nltk.word_tokenize(corpus)
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                                key = operator.itemgetter(1),
                                reverse = True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq)
                    for text, freq in sorted_ngrams]

    return sorted_ngrams

def getPOL():

    f=open('../fullStrengthLexicon.txt',encoding = "latin-1")
    f2=open('../mediumStrengthLexicon.txt',encoding = "latin-1")
    xml=f.read(); 
    f.close()
    xml2=f2.read(); 
    f2.close()
    tabla = {}

    xml = xml.split('\n')
    xml2 = xml2.split('\n')
    worlds = list()
 
    for x in xml:
        y = list()
        x = x.split('\t')
        word = delete_diacritic_marks(x[0])
        score = x[-1]
        tabla[word] = score
    
    for x in xml2:
        y = list()
        x = x.split('\t')
        word = delete_diacritic_marks(x[0])
        score = x[-1]
        tabla[word] = score

    return tabla

def delete_diacritic_marks(word):
    import re
    word = word.lower()
    word = re.sub('[á]', 'a', word)
    word = re.sub('[é]', 'e', word)
    word = re.sub('[í]', 'i', word)
    word = re.sub('[ó]', 'o', word)
    word = re.sub('[ú]', 'u', word)
    word = re.sub('[ñ]', 'n', word)
    return word


if __name__ == '__main__':
    taggerPath = 'tagger.pkl'
    tagger = []
    lemma = {}
    exclude = set(string.punctuation)
    exclude.update(['…','¿', '¡', '``'])

    # Check if the file tagger.pkl exists
    # if so load tagger, if not create a tagger
    if os.path.isfile(taggerPath):
        # print('Loading Tagger')
        tagger = loadPkl(taggerPath)
    else:
        print('Initialiazing Tagger')
        spanishTags = nltk.corpus.cess_esp.tagged_sents()
        tagger = nltk.UnigramTagger(spanishTags)
        savePkl(tagger, taggerPath)

    # list of texts, each text is a string (a sms)
    reviews, classification, reviewC, corpus = readTagMessages(
        '../../Corpus/SFU_Spanish_Review_Corpus/moviles/', tagger)

    frases = ['manos libres', 'pantalla', 'bateria', 'calidad', 'memoria', 'precio', 'juegos', 'camara']
    positivas = []
    negativas = []
    for e, val in enumerate(frases):
        positivas.append(0)
        negativas.append(0)

    sentences = list()

    for p in reviewC:
        sentences.append(p.split('.'))

    # print(sentences)
    tabla = getPOL()
    for sentence in sentences:
        for x in sentence:
            posi = 0
            nega = 0
            # print(x)
            words=cleanText(x)
            for word in words.split(' '):
                word=word.lower()
                words=delete_diacritic_marks(word)
                if word in tabla:
                    if tabla[word] == 'pos':
                        posi += 1
                    elif tabla[word] == 'neg':
                        nega += 1
            for index, val in enumerate(frases):
                val = nltk.word_tokenize(val)
                if val[0] in x:
                    positivas[index] += posi
                    negativas[index] += nega

    for index, val in enumerate(frases):
        pos = positivas[index]/(positivas[index]+negativas[index])
        neg = negativas[index]/(positivas[index]+negativas[index])
        print('{} {} {}'.format(val, pos, neg))

    corpus = nltk.sent_tokenize(corpus, 'spanish')
    for sent in corpus:
        for word in nltk.word_tokenize(sent, 'spanish'):
            pass

    # nouns = Counter()
    # for i, review in enumerate(reviews):
    #     # This is a counter using buckets
    #     for word, tag in review:
    #         if tag.startswith('n'):
    #             nouns[delete_diacritic_marks(word)] += 1
    # pprint(nouns.most_common(40))
