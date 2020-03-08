import math
import nltk
import operator
from bs4 import BeautifulSoup
import re

def writeList(myList, fname):
    f = open(fname, "w", encoding="utf-8")#open a UTF-8 text file for writing
    for i in range(len(myList)):
        f.write(str(myList[i])+'\n') 
    f.close()
    # print('Message of writeList(myList, fname): '+fname+' has '+str(len(myList))+' lines.\n')

def get_vocabulary(alist):
    vocabulary=sorted(set(alist))
    # print('There are', len( vocabulary), 'words in  vocabulary.\n')
    return vocabulary

def delete_stopwords(clean_tokens):
    stopword_list = nltk.corpus.stopwords.words('spanish')
    filtered_tokens = [token for token in clean_tokens if token not in stopword_list]
    return filtered_tokens

def get_clean_tokens(raw_tokens):
    clean_tokens=[]
    for tok in raw_tokens:
        t=[]
        for char in tok: 
            if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÑ]', char):#for Spanish alphabet
                t.append(char)
        letterToken=''.join(t)
        if letterToken !='':
            clean_tokens.append(letterToken)
    # print('There are', len(clean_tokens), 'clean tokens.\n')
    return clean_tokens

def get_raw_tokens(text_string):
    raw_tokens=nltk.Text(nltk.word_tokenize(text_string))
    # print('There are', len(raw_tokens), 'raw tokens.\n')
    return raw_tokens

def get_text_string(fname): 
    f=open(fname, encoding='latin-1')
    text_string=f.read()
    f.close()
    soup = BeautifulSoup(text_string, 'lxml')
    text_string = soup.get_text()
    text_string = text_string.replace('\x97', ' ')
    text_string=text_string.lower()
    # print('The text in', fname, 'has', len(text_string), 'characters.\n')
    return text_string

def conditional_entropy(pW1_1,pW2_1,pW1_1W2_1):
    pW2_0 = 1-pW2_1
    pW1_1W2_0 = pW1_1 - pW1_1W2_1
    pW1_0W2_0 = pW2_0 - pW1_1W2_0
    pW1_0W2_1 = pW2_1 - pW1_1W2_1
    if pW1_0W2_0>0 and pW1_0W2_1>0 and pW1_1W2_0>0 and pW1_1W2_1>0:
        condEntropy = (pW1_0W2_0*math.log(pW2_0/pW1_0W2_0,2))+\
                    (pW1_1W2_0*math.log(pW2_0/pW1_1W2_0,2))+\
                    (pW1_0W2_1*math.log(pW2_1/pW1_0W2_1,2))+\
                    (pW1_1W2_1*math.log(pW2_1/pW1_1W2_1,2))
    else:
        condEntropy = 0
    return condEntropy
    
def entropy2(pW1_1,pW2_1,pW1_1W2_1):
    pW2_0 = 1-pW2_1
    pW1_0 = 1-pW1_1
    pW1_1W2_0 = pW1_1 - pW1_1W2_1
    pW1_0W2_0 = pW2_0 - pW1_1W2_0
    pW1_0W2_1 = pW2_1 - pW1_1W2_1
    if pW1_0W2_0>0 and pW1_0W2_1>0 and pW1_1W2_0>0 and pW1_1W2_1>0:
        condEntropy = (pW1_0W2_0*math.log(pW1_0W2_0/(pW1_0*pW2_0),2))+\
                    (pW1_0W2_1*math.log(pW1_0W2_1/(pW1_0*pW2_1),2))+\
                    (pW1_1W2_0*math.log(pW1_1W2_0/(pW1_1*pW2_0),2))+\
                    (pW1_1W2_1*math.log(pW1_1W2_1/(pW1_1*pW2_1),2))
    else:
        condEntropy = 0
    return condEntropy

def get_sentences(text_string):
    sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(text_string)
    # print('There are',len(sentences),'sentences')
    return sentences
    
if __name__=='__main__':
    fname = '../../Corpus/e960401.htm'
    text_string = get_text_string(fname)
    raw_tokens = get_raw_tokens(text_string)
    clean_tokens = get_clean_tokens(raw_tokens)
    tokens_without_stopwords = delete_stopwords(clean_tokens)
    vocabulary = get_vocabulary(tokens_without_stopwords)
    
    sentences = get_sentences(text_string)
    N = len(sentences)
    
    pW2_1 = []
    for W2 in vocabulary:
        freq = 0
        for sent in sentences:
            if W2 in sent:
                freq += 1
        pW2_1.append((freq+0.5) / (N+1))
        
    W1 = 'empresa'
    
    index = vocabulary.index(W1)
    
    pW1_1 = pW2_1[index]
    
    pW1_1W2_1 = []
    for W2 in vocabulary:
        freq = 0
        for sent in sentences:
            if W1 in sent and W2 in sent:
                freq += 1
        pW1_1W2_1.append((freq+0.25) / (N+1))
        
    cond_entropy = {}
    entropy2Aux = {}
    for i in range(len(vocabulary)):
        cond_ent = conditional_entropy(pW1_1,pW2_1[i],pW1_1W2_1[i])
        cond_ent2 = entropy2(pW1_1,pW2_1[i],pW1_1W2_1[i])
        if cond_ent:
            cond_entropy[vocabulary[i]] = cond_ent
            entropy2Aux[vocabulary[i]] = cond_ent2
            
    sorted_entropy = sorted(cond_entropy.items(),key=operator.itemgetter(1))
    sorted_entropy = sorted_entropy[:50]
    sorted_entropy2 = sorted(entropy2Aux.items(),key=operator.itemgetter(1), reverse=True)
    sorted_entropy2 = sorted_entropy2[:50]
        # print('Entropias entre las palabras')
    print('Entropias entre las palabras')
    # for i in sorted_entropy:  
    #   print(i)
    for i in sorted_entropy2[:50]:  
        print(i[0], i[1])
    writeList(sorted_entropy2,'empresa_cond_entropy2.txt')
    
