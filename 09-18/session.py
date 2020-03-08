# import mmap
import re
import math
import string
# from functools import reduce
# import numpy as np
import nltk
# This library is for a HTML Parser
from bs4 import BeautifulSoup


def remove_characters_after_tokenization(tokens):
    '''Remove special characters, we receive tokens'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token)
                                    for token in tokens])
    return list(filtered_tokens)


def save_words(name, words):
    '''Save a file with a list of words, each word in a new line'''
    with open(name, "w") as f:
        for w in words:
            f.write(w+"\n")


def getText(filepath, encoding='utf-8'):
    # Read the hole file
    # The test has a sample of the hole file, only one paragraph
    text = ""
    with open(corpus_root+"/"+article_name, encoding=encoding) as f:
        text = f.read()
    return text


def tokenizeHTML(text):
    soup = BeautifulSoup(text, 'lxml')
    parsedText = soup.get_text()
    parsedText = parsedText.replace('\x97', ' ')
    for c in string.punctuation:
        parsedText.replace(c, ' ')
    return nltk.Text(nltk.word_tokenize(parsedText))


def cleanTokens(tokens, sw=True):
    # Para usar stopwords sw = False
    stopwords = nltk.corpus.stopwords.words('spanish')
    if sw:
        cleanedTokens = remove_characters_after_tokenization([
            t.lower() for t in tokens
            if t.isalpha() and t.lower() not in stopwords
        ])
    else:
        cleanedTokens = remove_characters_after_tokenization([
             t.lower() for t in tokens if t.isalpha()])
    return nltk.Text(cleanedTokens)


lemma = {}


def lemmanize(tokens):
    global lemma
    if len(lemma) == 0:
        print("lemma init")
    # lemma = {}
        with open('../generate.txt', encoding='latin-1') as f:
            for l in f:
                s = l.replace('#', '')
                words = s.split()
                if len(words) > 2:
                    lemma[words[0]] = words[-1]

    lemmanized = []
    unkown = []
    for w in tokens:
        try:
            lemmanized.append(lemma[w])
        except KeyError:
            lemmanized.append(w)
            unkown.append(w)
    return lemmanized, unkown


def conditionalEntropy(pW1_1, pW2_1, pW1_1W2_1):
    pW2_0 = 1 - pW2_1
    pW1_1W2_0 = pW1_1 - pW1_1W2_1
    pW1_0W2_0 = pW2_0 - pW1_1W2_0
    pW1_0W2_1 = pW2_1 - pW1_1W2_1

    if pW1_0W2_0 > 0 and pW1_0W2_1 > 0 and pW1_1W2_0 > 0 and pW1_1W2_1 > 0:
        return (pW1_0W2_0*math.log(pW2_0/pW1_0W2_0, 2)) + \
                (pW1_1W2_0*math.log(pW2_0/pW1_0W2_0, 2)) + \
                (pW1_0W2_1*math.log(pW2_1/pW1_0W2_1, 2)) + \
                (pW1_1W2_1*math.log(pW2_1/pW1_1W2_1, 2))
    else:
        return 0


if __name__ == "__main__":
    corpus_root = '../../Corpus'
    article_name = 'e960401.htm'
    text = getText(corpus_root+article_name, "latin-1")
    articles = text.split('<h3>')[1:]
    print("There are {} articles in the file {}.".format(
        len(articles), article_name))
    for i, article in enumerate(articles):
        body = article.split('</h3>')
        title, content = body[0], body[1]
        print("\n\n\nThe title of the next article is: ", " ".join(
            cleanTokens(tokenizeHTML(title), False)))
        rawTokens = tokenizeHTML(article)
        # print(rawTokens[:10])
        cleanedTokens = cleanTokens(rawTokens)
        print(i, " lemmanizing...")
        lemmanized, unkown = lemmanize(cleanedTokens)
        # print(lemmanized[:20])
        # vocabulary = sorted(set(lemmanized))
        vocabulary = sorted(set(cleanedTokens))
        # print(vocabulary[:10])
        print("This article has ", len(lemmanized), " tokens lemmanized")
        print("This article has ", len(vocabulary), " vocabularyn")

        freq = [[w, lemmanized.count(w)] for w in vocabulary]

        for w, c in sorted(freq, key=lambda e: e[1], reverse=True)[:5]:
            print(w, c)
