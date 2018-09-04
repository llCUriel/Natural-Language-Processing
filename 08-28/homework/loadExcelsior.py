# Reading the e960401 file for basic manipulation

import nltk
from nltk.corpus import PlaintextCorpusReader

corpus_root = '../../../Corpus'

excelsior = PlaintextCorpusReader(corpus_root, '.*\.txt')
print("Available articles ", excelsior.fileids())

article_name = 'e960401.txt'
article = excelsior.words(fileids=article_name)
article_lower = [w.lower() for w in article]

print(article_name, " has ", len(article_lower), " tokens.")

vocabulary = sorted(set(article_lower))
print(vocabulary)
print(article_name, " has a vocabulary length of ", len(vocabulary), ".")
text = nltk.Text(article_lower)
# text.concordance('empresa')

bag = []
for cl in text.concordance_list('empresa'):
    left = list(cl[0][-4:])
    right = list(cl[2][-4:])
    bag += left
    bag += right

print("The bag of words of 'empresa' is: ", bag)

print("Words similar to 'empresa': ", text.similar('empresa'))
