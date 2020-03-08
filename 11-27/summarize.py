import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re
import os
# import resource
import string
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import nltk
import operator
import networkx
from bs4 import BeautifulSoup
from pprint import pprint
import numpy as np


def savePkl(tagger, filename="name.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(tagger, f, -1)


def loadPkl(filename="name.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def lemmanize(tokens, taggin=False):
    global lemma
    global tagger

    if len(lemma) == 0:
        # print("lemma init")
        with open('../generate.txt', encoding='latin-1') as f:
            for l in f:
                s = l.replace('#', '')
                words = s.split()
                if len(words) > 2:
                    if taggin:
                        lemma[words[0]] = words[-1], str(words[-2]).lower()
                    else:
                        lemma[words[0]] = words[-1]

    lemmanized = []
    unknown = []
    for w in tokens:
        try:
            lemmanized.append(lemma[w])
        except KeyError:
            if taggin:
                lemmanized.append((w, 'nounArtificial'))
                unknown.append((w, 'nounArtificial'))
            else:
                lemmanized.append(w)
                unknown.append(w)
    return lemmanized, unknown

def delete_diacritic_marks(word):
    import re
    word = word.lower()
    word = re.sub('[á]', 'a', word)
    word = re.sub('[é]', 'e', word)
    word = re.sub('[í]', 'i', word)
    word = re.sub('[ó]', 'o', word)
    word = re.sub('[ú]', 'u', word)
    word = re.sub('[ñ]', 'n', word)
    return word


def removeSpecialCharacters(tokens):
    '''Remove special characters, we receive tokens'''
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filteredTokens = filter(None, [pattern.sub(' ', token)
                                   for token in tokens])

    return list(filteredTokens)


def cleanText(text, lemmanized=False, cleaningLevel=1, taggin=False):
    ''' A string is cleaned, depending on the level of cleaning:
            0 -> raw string
            1 - > lower case, special characters removed
            2 - > Stopwords are removed
        the cleaned string is return
    '''
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('spanish')

    if cleaningLevel == 0:
        cleanedTokens = tokens
    elif cleaningLevel == 1:
        cleanedTokens = removeSpecialCharacters([
            t.lower() for t in tokens if t.isalpha()
        ])
    elif cleaningLevel == 2:  # Without stopwords
        cleanedTokens = removeSpecialCharacters([
            t.lower() for t in tokens
            if t.isalpha() and t.lower() not in stopwords
        ])
    if taggin:
        sentences = nltk.sent_tokenize(text, language='spanish')
        tagged = []
        tokens = []
        # Tags using the tagger, as a fallback,
        # it uses the tag from generator.txt dictonary
        for sentence in sentences:
            taggedTokens = tagger.tag(nltk.word_tokenize(sentence, 'spanish'))
            for token, tag in taggedTokens:
                token = str(token).lower()
                tag = str(tag).lower()
                # When the token was a punctuation character token is empty
                if token.isalpha() and token not in stopwords:
                    lemmanized, unknown = lemmanize([token], taggin)
                    # If it's a token that isn't on the dictonary
                    if len(unknown) == 1:
                        lemmanized[0] = lemmanized[0], 'nounArtificial'
                    else:
                        tag = lemmanized[0][1]
                    tokens.append(token)
                    tagged.append((token, tag))
        return tokens, tagged
    else:
        lemmanizedTokens, unkown = lemmanize(cleanedTokens, taggin)
        cleanedText = ' '.join(cleanedTokens)
        return cleanedText


def readTagMessages(path, tagger=None, cleaningLevel=1, lemmanized=False,
                    encoding='latin-1'):
    reviews = []
    tagged = []
    classifications = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith('txt'):
            with open(path+filename, encoding=encoding) as f:
                review = f.read().strip()
                x = review
                review, tags = cleanText(review, lemmanized,
                                         cleaningLevel, True)
                # review = get_top_ngrams(review)
                classification = filename.split('.')[0].split('_')
                tagged.append(tags)
                reviews.append(x)
                classifications.append((filename, classification))

    return tagged, classifications, reviews


def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] for index in range(n)])

def get_top_ngrams(tokens, ngram_val=1, limit=5):
    # cleanText(corpus)
    # tokens = nltk.word_tokenize(corpus)
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                                key = operator.itemgetter(1),
                                reverse = True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq)
                    for text, freq in sorted_ngrams]

    return sorted_ngrams

def getPOL():

    f=open('../fullStrengthLexicon.txt',encoding = "latin-1")
    f2=open('../mediumStrengthLexicon.txt',encoding = "latin-1")
    xml=f.read();
    f.close()
    xml2=f2.read();
    f2.close()
    tabla = {}

    xml = xml.split('\n')
    xml2 = xml2.split('\n')
    worlds = list()

    for x in xml:
        y = list()
        x = x.split('\t')
        word = delete_diacritic_marks(x[0])
        score = x[2]
        tabla[word] = score

    for x in xml2:
        y = list()
        x = x.split('\t')
        word = delete_diacritic_marks(x[0])
        score = x[2]
        tabla[word] = score

    return tabla

def delete_diacritic_marks(word):
    import re
    word = word.lower()
    word = re.sub('[á]', 'a', word)
    word = re.sub('[é]', 'e', word)
    word = re.sub('[í]', 'i', word)
    word = re.sub('[ó]', 'o', word)
    word = re.sub('[ú]', 'u', word)
    word = re.sub('[ñ]', 'n', word)
    return word

def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii','ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def build_feature_matrix(documents, feature_type='frequency'):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1,ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1,ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1,ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values:'binary', 'frequency', 'tfidf'")
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    return vectorizer, feature_matrix

def textrank_text_summarizer(documents, num_sentences=2,feature_type='frequency'):
    vec, dt_matrix = build_feature_matrix(documents,feature_type='tfidf')

    similarity_matrix = (dt_matrix * dt_matrix.T)

    print('\n\tThis is a similarity matriz:\n')
    print(np.round(similarity_matrix.todense(),2))

    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    networkx.draw_networkx(similarity_graph)

    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()),reverse=True)

    print('\n\tThese are ranked sentences:\n')
    for r in ranked_sentences:
        print(r)

    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]
    top_sentence_indices.sort()
    print('\n\tThese are top sentence indices:\n')
    print(top_sentence_indices)

    print('\n\t Summary generated by the TextRank algorithm')
    print('\tThis summary has',len(top_sentence_indices),'sentences: \n')
    for index in top_sentence_indices:
        print (sentences[index])

    plt.subplot(111)
    print('\n\tText similarity graph:\n')
    plt.show()

if __name__ == '__main__':
    taggerPath = 'tagger.pkl'
    tagger = []
    lemma = {}
    exclude = set(string.punctuation)
    exclude.update(['…','¿', '¡', '``'])

    # Check if the file tagger.pkl exists
    # if so load tagger, if not create a tagger
    if os.path.isfile(taggerPath):
        # print('Loading Tagger')
        tagger = loadPkl(taggerPath)
    else:
        print('Initialiazing Tagger')
        spanishTags = nltk.corpus.cess_esp.tagged_sents()
        tagger = nltk.UnigramTagger(spanishTags)
        savePkl(tagger, taggerPath)

    # list of texts, each text is a string (a sms)
    reviews, classification, reviewC = readTagMessages(
        '../../Corpus/SFU_Spanish_Review_Corpus/moviles/', tagger)

    sentences = list()

    for p in reviewC:
        sentences.append(p.split('.'))

    # print(sentences)
    tabla = getPOL()
    tablaPosi = {}
    tablaNega = {}
    sentencesWi = ''
    for sentence in sentences:
        tablaPosiTEMP = {}
        tablaNegaTEMP = {}
        sentenceTEMP = ''
        flag = None
        for x in sentence:
            # print(x)
            words=cleanText(x)
            for word in words.split(' '):
                word=word.lower()
                words=delete_diacritic_marks(word)
                sentenceTEMP += ' '
                sentenceTEMP += word
                if word == 'bateria':
                    flag = True
        if flag:
            sentencesWi += '.'
            sentencesWi += sentenceTEMP
    print(sentencesWi)
    sentences = parse_document(sentencesWi)
    total_sentences = len(sentences)
    print(sentences)
    print('\n\tTotal Sentences in Document:',total_sentences)
    textrank_text_summarizer(sentences,num_sentences=3,feature_type='tfidf')
