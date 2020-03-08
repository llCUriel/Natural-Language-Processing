# Reading the e960401 file for basic manipulation

import re
import math
import string
from functools import reduce
import numpy as np
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


def get_context(tokens, word, window=8):
    '''Get the context or bag of words of a given word inside a text'''
    bag = []
    cl = tokens.concordance_list(word, lines=tokens.count(word))
    # tokens.concordance(word)
    for c in cl:
        left = list(c[0][-window//2:])
        right = list(c[2][:window//2])
        bag += left
        bag += right
    return list(bag)


corpus_root = '../../Corpus'
article_name = 'e960401.htm'
text = ""
# Read the hole file
# The test has a sample of the hole file, only one paragraph
with open(corpus_root+"/"+article_name, encoding='latin-1') as f:
    text = f.read()

soup = BeautifulSoup(text, 'lxml')
parsedText = soup.get_text()
parsedText = parsedText.replace('\x97', ' ')
for c in string.punctuation:
    parsedText.replace(c, ' ')

tokens = nltk.Text(nltk.word_tokenize(parsedText))
print("Amount of raw tokens ", len(tokens))
# Para usar stopwords
# tokens = remove_characters_after_tokenization([
#      t.lower() for t in tokens if t.isalpha()])
# Para quitar stopwords
stopwords = nltk.corpus.stopwords.words('spanish') + \
        ['http', 'lunes', 'martes', 'abril']
tokens = remove_characters_after_tokenization([
    t.lower() for t in tokens if t.isalpha() and t.lower() not in stopwords])
tokens = nltk.Text(tokens)
# save_words('tokens.txt', tokens)


tokens.collocations()
