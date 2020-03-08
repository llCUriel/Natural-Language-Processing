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

# Get the context, and compute the similarity between search words
# search_words = ['empresa', 'agua', 'compañía', 'empresa']
search_words = ['empresa'] + vocabulary
vectors = []
bags = []
# K = [1.2, 2] 
k = 1.2
for w in search_words:
    bag = get_context(tokens, w, 8)
    # print("The bag of words of '", w, "' is: ", bag[:8])
    # save_words('context-'+w+'.txt', bag)
    # using term frequency
    # vector = [np.array([bag.count(t) for t in vocabulary]), 0, w]
    # EOWC using probabilistic term
    # vector = [np.array([bag.count(t)/len(bag) for t in vocabulary]), 0, w]
    # Using TF Transformation: BM25
    # vector = [np.array([((1+bag.count(t))*k) / (bag.count(t)+k)
    #           for t in vocabulary]), 0, w]
    # Inverse Document Frequency First Part Get all bags of term frequency
    # vector = [np.array([bag.count(t) for t in vocabulary]), 0, w]
    # print("The vector of ", w, " is ", vector)

    # Using TF*IDF
    vector = [np.array([((1+bag.count(t))*k) / (bag.count(t)+k)
              for t in vocabulary]), 0, w]
    vectors.append(vector)
    bags.append(bag)
    # Check if we count all items correctly
    # assert len(bag) == reduce((lambda x, y: x+y), vector)

# for i in range(1, len(vectors)):
#   # k = 0
#   # for j in range(1, len(vectors)):
#       # if vectors[i][2] in bags[j]:
#           # k += 1

#   # for j in range(len(vectors[i][3])):
#       # vectors[i][3][j] = math.log10(len(vocabulary)+1 / k)
for i in range(1, len(vectors)):
    itf = [1 for j in range(len(vocabulary))]
    for j in range(len(vocabulary)):
        itf[j] = math.log10((len(vocabulary)+1) / len(set(bags[j])))
    # for j in range(len(vocabulary)):
        # print(vectors[i][0][j], itf[j], vectors[i][0][j]*itf[j])

    vectors[i][0] = vectors[i][0]*itf

# print("Vectors are ", vectors)

for i in range(1):
    for j in range(i+1, len(search_words)):
        dot = np.dot(vectors[i][0], vectors[j][0])
        mag1 = np.sqrt(vectors[i][0].dot(vectors[i][0]))
        mag2 = np.sqrt(vectors[j][0].dot(vectors[j][0]))
        sim = dot/(mag1*mag2)
        vectors[j][1] = sim

vectors = sorted(vectors, key=lambda e: e[1], reverse=True)

similarWords = []
for i in range(20):
        r = "sim({}, {}) = {}".format(search_words[0], vectors[i][2],
                                      vectors[i][1])
        similarWords.append(vectors[i][1])
        print(r)
