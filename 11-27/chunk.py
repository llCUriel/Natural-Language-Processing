import re
import os
# import resource
import string
from collections import Counter

import pickle
import nltk

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from pprint import pprint
import numpy as np
import networkx
from gensim.summarization.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer


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
        print("lemma init")
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


def deleteDiacriticMarks(word):
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
        cleanedTokens = ' '.join(tokens)
        return cleanedTokens, tagged
    else:
        lemmanizedTokens, unkown = lemmanize(cleanedTokens, taggin)
        cleanedText = ' '.join(cleanedTokens)
        return cleanedText


def readTagMessages(path, tagger=None, cleaningLevel=1, lemmanized=False,
                    encoding='latin-1'):
    reviews = []
    tagged = []
    classifications = []
    sentences = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith('txt'):
            with open(path+filename, encoding=encoding) as f:
                print('Review ', filename, ' said: ')
                review = f.read().strip()
                # OriginalReviewLength
                osl = nltk.sent_tokenize(review, 'spanish')
                for sent in osl:
                    sentences.append(sent.strip())
                if len(osl) > 5:
                    summary = summarize(review)
                    # Summary Review Length
                    ssl = nltk.sent_tokenize(summary, 'spanish')
                    print('Original has ', len(osl), ' sentences.')
                    print('Summary has ', len(ssl), ' sentences.')
                    pprint(summary)
                else:
                    print(filename, ' cannot be summarized.')
                review, tags = cleanText(review, lemmanized,
                                         cleaningLevel, True)
                classification = filename.split('.')[0].split('_')
                tagged.append(tags)
                reviews.append(review)
                classifications.append((filename, classification))
                print()
                print('-'*50)

    return tagged, classifications, reviews, sentences

def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] for index in range(n)])


def get_top_ngrams(corpus, ngram_val=1, limit=5):
    tokens = nltk.word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    # pprint(ngrams_freq_dist.items())
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                              key=lambda e: e[1], reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq)
                     for text, freq in sorted_ngrams]
    return sorted_ngrams


if __name__ == '__main__':
    taggerPath = 'tagger.pkl'
    tagger = []
    lemma = {}

    # Check if the file tagger.pkl exists
    # if so load tagger, if not create a tagger
    if os.path.isfile(taggerPath):
        print('Loading Tagger')
        tagger = loadPkl(taggerPath)
    else:
        print('Initialiazing Tagger')
        spanishTags = nltk.corpus.cess_esp.tagged_sents()
        tagger = nltk.UnigramTagger(spanishTags)
        savePkl(tagger, taggerPath)

    sentence = "Los estudiantes de ESCOM ganaron el premio Nobel."

    taggedTokens = tagger.tag(nltk.word_tokenize(sentence, 'spanish'))
    tokens = []
    stopwords = nltk.corpus.stopwords.words('spanish')
    for token, tag in taggedTokens:
        token = str(token).lower()
        tag = str(tag).lower()
        # When the token was a punctuation character token is empty
        if token.isalpha():
            lemmanized, unknown = lemmanize([token], True)
            # If it's a token that isn't on the dictonary
            if len(unknown) == 1:
                lemmanized[0] = lemmanized[0], 'nounArtificial'
            else:
                tag = lemmanized[0][1]
            if tag == 'none':
                tag = 'noun'
            tokens.append((token, tag))
    pprint(tokens)
    grammar = "NP: {<td.*>?<n.*><s.*>?<n.*>?}"

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(tokens)
    print(result)
    result.draw()

    sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
                ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),
                ("the", "DT"), ("cat", "NN")]

    grammar = "NP: {<DT>?<JJ>*<NN>}"

    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sentence)
    print(result)
    result.draw()
