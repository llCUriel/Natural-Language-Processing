# import mmap
import os.path
import re
import resource
# import math
import string
from pickle import dump, load
import gensim
from gensim import corpora
# from functools import reduce
# import numpy as np
import nltk
from pprint import pprint
# This library is for a HTML Parser
from bs4 import BeautifulSoup

import logging


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


def tokenizeHTML(text, sent=False):
    soup = BeautifulSoup(text, 'lxml')
    parsedText = soup.get_text()
    parsedText = parsedText.replace('\x97', ' ')
    for c in string.punctuation:
        parsedText.replace(c, ' ')
    return nltk.Text(nltk.word_tokenize(parsedText)) if not sent \
        else nltk.Text(nltk.sent_tokenize(parsedText))


def cleanTokens(tokens, sw=True):
    # Para usar stopwords sw = False
    stopwords = nltk.corpus.stopwords.words('spanish') + ['http']
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
                    lemma[words[0]] = words[-1], str(words[-2]).lower()

    lemmanized = []
    unknown = []
    for w in tokens:
        try:
            lemmanized.append(lemma[w])
        except KeyError:
            lemmanized.append(w)
            unknown.append(w)
    return lemmanized, unknown


def saveTagger(tagger, filename="tagger.pkl"):
    with open(filename, "wb") as f:
        dump(tagger, f, -1)


def loadTagger(filename="tagger.pkl"):
    ''' Check if the file tagger.pkl exists
     if so load tagger, if not create a tagger.'''
    if os.path.isfile(taggerPath):
        with open(filename, "rb") as f:
            return load(f)
    else:
        spanishTags = nltk.corpus.cess_esp.tagged_sents()
        tagger = nltk.UnigramTagger(spanishTags)
        saveTagger(tagger, filename)
        return tagger


def getArticles(text):
    ''' Split the text into artcles '''
    # The first artcle is removed because is the html header
    global tokensByArticle
    articles = text.split('<h3>')[1:]
    print("There are {} articles in the file {}.".format(
        len(articles), article_name))

    for article in articles:
        tokensArticle = []
        # Print the title of the article
        # body = article.split('</h3>')
        # title, content = body[0], body[1]
        # print("\n\nThe title of the next article is: ", " ".join(
        #     cleanTokens(tokenizeHTML(title), False)))
        # Split into sentences
        sentences = tokenizeHTML(article, True)
        # print(sentences[:2])
        for sentence in sentences:
            # Tag the tokens of each sentence
            words = tagger.tag(tokenizeHTML(sentence))
            for word, tag in words:
                tag = str(tag).lower()
                # Lower and remove the stopwords from the tokens
                w = cleanTokens([word])
                # When the token was a punctuation character w is empty
                if len(w) > 0:
                    # print(w[0], str(tag).lower())
                    lemmanized, unkown = lemmanize([w[0]])
                    if len(unkown) == 1:
                        lemmanized[0] = lemmanized[0], '0'
                    tokensArticle.append(lemmanized[0][0])
        tokensByArticle.append(list(tokensArticle))
        # print('This article has ', len(tokens))
        # print('This article has ', len(vocabulary), ' distinct tokens.')
        yield tokensArticle


if __name__ == "__main__":
    corpus_root = '../../Corpus'
    article_name = 'e960401.htm'
    taggerPath = 'tagger.pkl'
    tokensByArticle = []
    tagger = loadTagger(taggerPath)

    text = getText(corpus_root+article_name, "latin-1")
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
    #                   # level=logging.INFO)
    '''Creating the term dictionary of our courpus,
    where every unique term is assigned an index.'''
    dictionary = corpora.Dictionary(getArticles(text))
    dictionary.save('dictionary.dict')
    dictionary.save_as_text('dictionary.txt')

    '''Converting list of documents (corpus) into
    Document Term Matrix using dictionary prepared above.'''
    docTermMatrix = [dictionary.doc2bow(doc) for doc in tokensByArticle]

    '''Creating the object for LDA model using gensim library'''
    Lda = gensim.models.ldamodel.LdaModel

    '''Running and Trainign LDA model on the document term matrix.'''
    ldamodel = Lda(docTermMatrix, num_topics=5, id2word=dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=5, num_words=8))
    print(dictionary)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
