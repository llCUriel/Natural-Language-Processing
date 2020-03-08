# coding: utf-8
# Reading the e960401 file for basic manipulation
import mmap
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
stopwords = nltk.corpus.stopwords.words('spanish')
tokens = remove_characters_after_tokenization([
    t.lower() for t in tokens if t.isalpha() and t.lower() not in stopwords])
# Para usar stopwords
# tokens = remove_characters_after_tokenization([
#      t.lower() for t in tokens if t.isalpha()])
tokens = nltk.Text(tokens)
save_words('tokens.txt', tokens)

print("Amount of clean tokens without stopwords ", len(tokens))
vocabulary = sorted(set(t.lower() for t in tokens if t.isalpha()))
save_words('vocabulary.txt', vocabulary)
print("Vocabulary lenght = ", len(vocabulary))

print(vocabulary[:20])
print(vocabulary[-20:])

# with open('../generate.txt', 'rb', 0) as file,\
#       # mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
#   # index = s.find(b'zigzag')
#   # if index != -1:
#       # print(s[index:index+len('zigzag')], 'true', index)

lemma = {}
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
print("Number of tokens lemmanized ", len(lemmanized))
print("Number of unkown tokens lemmanized ", len(unkown))
save_words('lemmanized.txt', lemmanized)
save_words('unkown.txt', sorted(unkown))
save_words('unkown-set.txt', sorted(set(unkown)))
del lemmanized
del unkown
