# coding: utf-8
from nltk.corpus import brown
brown.categories()
print("Numero de categorias ",len(brown.categories()))
print("Numero de archivos ",len(brown.fileids()))
print("Numero de caracteres en cr09 ",len(brown.raw(fileids=['cr09'])))
print("Numero de palabras en cr09 ",len(brown.words(fileids=['cr09'])))
print("Numero de oraciones en cr09 ",len(brown.sents(fileids=['cr09'])))
get_ipython().run_line_magic('save', 'cr09')
get_ipython().run_line_magic('save', 'cr09 1-7')
r = brown.words(categories=['romance'])
import nltk
fdist = nltk.FreqDist(w.lower() for w in r)
words = ['love', 'hate
words = ['love', 'hate']
for w in words:
    print(w + ": ", fdist[m])
    
for w in words:
    print(w + ": ", fdist[w])
    
    
from nltk.corpus import PlaintextCorpusReader
categories = ['news', 'romance', 'humor']
words = ['love', 'hate
words = ['love', 'hate']
words = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
categories = ['news', 'romance']
cfd = nltk.ConditionalFreqDist(
    (categorie, word)
    for categorie in categories
    for word in brown.words(categories=categories)
    )
    
cfd.tabulate()
cfd.tabulate(conditions=categories)
cfd['monday']
cfd['news']
cfd['news']['monday']
cfd['news']['Monday']
for w in words:
    print(w + ": ", fdist[w])
    
    
for w in words:
    for c in categories:
        print(w + ": ", cfd[c][w]) 
           
words = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for w in words:
    for c in categories:
        print(w + ": ", cfd[c][w]) 
           
for c in categories:
    for w in words:
        print(cfd[c][w], ends=' ')
         
           
for c in categories:
    for w in words:
        print(cfd[c][w], endswith=' ')
        
         
           
for c in categories:
    for w in words:
        print(cfd[c][w], end=' ')
        
        
         
           
get_ipython().run_line_magic('save', 'session 1-37')
