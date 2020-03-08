# import mmap
import os.path
import re
import math
import string
from pickle import dump, load
# from functools import reduce
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


def saveTagger(tagger, filename="tagger.pkl"):
    with open(filename, "wb") as f:
        dump(tagger, f, -1)


def loadTagger(filename="tagger.pkl"):
    with open(filename, "rb") as f:
        return load(f)


if __name__ == "__main__":
    corpus_root = '../../Corpus'
    article_name = 'e960401.htm'
    taggerPath = 'tagger.pkl'
    tagger = []

    # Check if the file tagger.pkl exists
    # if so load tagger, if not create a tagger
    if os.path.isfile(taggerPath):
        tagger = loadTagger()
    else:
        spanishTags = nltk.corpus.cess_esp.tagged_sents()
        tagger = nltk.UnigramTagger(spanishTags)
        saveTagger(tagger)

    text = getText(corpus_root+article_name, "latin-1")
    # The first artcle is removed because is the html header
    articles = text.split('<h3>')[1:]
    print("There are {} articles in the file {}.".format(
        len(articles), article_name))

    # Split the text into artcles
    nouns = []
    tokens = []
    vocabularies = []
    tokensByArticle = []
    for article in articles:
        tokensArticle = []
        body = article.split('</h3>')
        # Print the title of the article
        title, content = body[0], body[1]
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
                    if tag == 'none':
                        tag = lemmanized[0][1]
                    if tag.startswith('n'):
                        # print(lemmanized[0][0])
                        nouns.append((lemmanized[0][0], tag))
                        # nouns.append(lemmanized[0][0])
                    tokens.append((lemmanized[0][0], tag))
                    tokensArticle.append(lemmanized[0][0])
                    # tokens.append(lemmanized[0][0])
        vocabulary = sorted(set([w for w in tokensArticle]))
        vocabularies.append([[v, 0] for v in vocabulary])
        # vocabulary = set([w for w in tokens])
        tokensByArticle.append(list(tokensArticle))
        # print('This article has ', len(tokens))
        # print('This article has ', len(vocabulary), ' distinct tokens.')
        del tokensArticle

    nounsSet = set(nouns)
    # save_words("nouns.txt", nounsSet)
    del nouns
    nouns = []
    # nounsCount = 0
    # for w in nounsSet:
    #   # # count = tokens.count(w)
    #   # count = np.sum([1 for token, tag, prob in tokens if token == w
    #                   # and tag.startswith('n')])
    #   # nounsCount += count
    #   # nouns.append([w, count])
    # nouns.sort(key=lambda e: e[1], reverse=True)
    for w, t in nounsSet:
        count = tokens.count((w, t))
        nouns.append([w, t, count])
    nouns.sort(key=lambda e: e[2], reverse=True)
    print("There are ", len(nouns), " distinct nouns.")
    # print("The nouns appeared ", nounsCount, " times.")
    # print([(n[0], n[1], n[1]/nounsCount) for n in nouns[:20]])
    print([(n[0], n[2]) for n in nouns[:20]])
    # print([n for n in nouns[:20]])
'''
    # print("There are ", len(tokens), " tokens.")
    # print(tokensByArticle[0])
    searchWords = ['inmigración', 'arquitectura', 'internet', 'astronauta',
                   'inflación']
    # searchWords = ['año', 'polítca', 'presidente', 'empresa',
    #              'millón']
    maxLen = max([len(w) for w in searchWords])
    p = '{:>'+str(maxLen)+'} '
    pat = len(searchWords)*p
    print(" article ", pat.format(*searchWords))
    for i, tokensArticle in enumerate(tokensByArticle):
        counts = []
        for w in searchWords:
            counts.append(tokensArticle.count(w))
        counts = np.array(counts)
        total = np.sum(counts)
        if total == 0:
            total = 1
        percentages = counts/total
        pattern = " ".join(len(percentages)*(((maxLen-8)*" ")+"{:>08.6f}",))
        print('{:^9d}'.format(i), pattern.format(*percentages))

    for i, vocabulary in enumerate(vocabularies):
        n = len(vocabulary)
        for i in range(n):
            count = np.sum([1 for token, tag, prob in tokensByArticle[i]
                            if token == w and tag.startswith('n')])
            tokens[i][2] = count/n
'''
