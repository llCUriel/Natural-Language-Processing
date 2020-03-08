import re
import os
# import resource
import string
import operator
from collections import Counter

import pickle
import nltk

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

    for filename in sorted(os.listdir(path)):
        if filename.endswith('txt'):
            with open(path+filename, encoding=encoding) as f:
                review = f.read().strip()
                review, tags = cleanText(review, lemmanized,
                                         cleaningLevel, True)
                classification = filename.split('.')[0].split('_')
                tagged.append(tags)
                reviews.append(review)
                classifications.append((filename, classification))

    return tagged, classifications, reviews

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

    # list of texts, each text is a string (a sms)
    reviews, classification, corpus, original = readTagMessages(
        '../../Corpus/SFU_Spanish_Review_Corpus/moviles/', tagger)
    nouns = Counter()
    for i, review in enumerate(reviews):
        # This is a counter using buckets
        for word, tag in review:
            if tag.startswith('n'):
                nouns[deleteDiacriticMarks(word)] += 1
        print(classification[i][0], ' has ', sum(nouns.values()), ' nouns.')
        print('And it has ', len(nouns), ' unique nouns.')
        print('-'*50)
    # pprint(nouns.most_common(25))

    ngrams = get_top_ngrams(' '.join(corpus), 2, 20)
    pprint(ngrams)

    corpus = nltk.sent_tokenize(original, 'spanish')
    for sentence in corpus:
        sent = ' '.join(nltk.word_tokenize(sentence))
        tokens = cleanText(sent, True, 2, False)
        if 'memoría' in tokens:




