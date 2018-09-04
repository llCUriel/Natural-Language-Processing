# coding: utf-8
from nltk.corpus import brown
brown.categories()
print("Numero de categorias ",len(brown.categories()))
print("Numero de archivos ",len(brown.fileids()))
print("Numero de caracteres en cr09 ",len(brown.raw(fileids=['cr09'])))
print("Numero de palabras en cr09 ",len(brown.words(fileids=['cr09'])))
print("Numero de oraciones en cr09 ",len(brown.sents(fileids=['cr09'])))
