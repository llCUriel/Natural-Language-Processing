# Reading the e960401 file for basic manipulation

import nltk
# This library is for a HTML Parser
from bs4 import BeautifulSoup

corpus_root = '../../../Corpus'


article_name = 'e960401.htm'
text = ""
# Read the hole file
# The test has a sample of the hole file, only one paragraph
with open(corpus_root+"/"+article_name, encoding='latin-1') as f:
    text = f.read()

soup = BeautifulSoup(text, 'lxml')
parsedText = soup.get_text()
parsedText = parsedText.replace('\x97', ' ')
tokens = nltk.Text(nltk.word_tokenize(parsedText))
print(len(tokens))
vocabulary = sorted(set(t.lower() for t in tokens if t.isalpha()))
print(len(vocabulary))
print(vocabulary[:20])
print(vocabulary[-20:])

# article_lower = [w.lower() for w in article]

# print(article_name, " has ", len(article_lower), " tokens.")

# vocabulary = sorted(set(article_lower))
# print(article_name, " has a vocabulary length of ", len(vocabulary), ".")
# text = nltk.Text(article_lower)
# # text.concordance('empresa')

# bag = []
# for cl in text.concordance_list('empresa'):
#     left = list(cl[0][-4:])
#     right = list(cl[2][-4:])
#     bag += left
#     bag += right

# print("The bag of words of 'empresa' is: ", bag)

# text.similar('empresa')
