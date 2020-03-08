import re
import string
from functools import reduce
import numpy as np
import nltk
from bs4 import BeautifulSoup
from pprint import pprint

def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences] 
    return word_tokens

def remove_characters_after_tokenization(tokens):
    '''Remove special characters, we receive tokens'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation+"[¡|¿|0|1|2|3|4|5|6|7|8|9|ö|\|/|Ö|-]"))) 
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens]) 
    return filtered_tokens

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('spanish')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens

def save_words(name, words):
    '''Save a file with a list of words, each word in a new line'''
    with open(name, "w") as f:
        for w in words:
            f.write(w+"\n")


def get_context(tokens, word, window=8):
    '''Get the context or bag of words of a given word inside a text'''
    bag = []
    for cl in tokens.concordance_list(word, lines=tokens.count(word)):
        left = list(cl[0][-window//2:])
        right = list(cl[2][:window//2])
        bag += left
        bag += right
    return list(bag)


corpus_root = '../../Corpus/'
article_name = 'e960401.htm'
text = ""
with open(corpus_root+"/"+article_name, encoding='latin-1') as f:
    text = f.read()

soup = BeautifulSoup(text, 'lxml')
parsedText = soup.get_text()
parsedText = parsedText.replace('\x97', ' ')
for c in string.punctuation:
    parsedText.replace(c, ' ')

tokens = nltk.Text(nltk.sent_tokenize(parsedText))
stopword_list = nltk.corpus.stopwords.words('spanish')
# print(stopword_list)
t1 = 0
t2 = 0

words = []
for token in tokens:
    x = nltk.word_tokenize(token)
    for y in x:
        if y.isalpha():
            # print(y)
            words.append(y)
            # print(y)
            t1 += 1
        break
print('Total de oraciones:', len(set(words)))
print('NE:', (62/len(set(words)))*100)
words = list(set(words))
pprint(words[:10])

del words
words = []
for token in tokens:
    x = nltk.word_tokenize(token)
    for y in x[1:]:
        if not y.islower() and not y.isupper() and y.isalpha():
            # print(y)
            words.append(y)
            t2 += 1
print('Total de palabras mayusculas:', len(set(words)))
print('NE:', (1335/len(set(words)))*100)
words = list(set(words))
pprint(words[:10])
