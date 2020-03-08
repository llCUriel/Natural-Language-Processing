import os   #for list_files
import nltk #for prepareData
import string
from bs4   import BeautifulSoup

def list_files(dir):
    fnames = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            fnames.append(os.path.join(name))
    return fnames

def list_files_xml(dir):
    fnames = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('xml'):
                fnames.append(os.path.join(name))
    return fnames

def list_files_pos(dir):
    '''For files .pos in the corpusCine
    '''
    fnames = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('pos') and 'summary' not in name:
                fnames.append(os.path.join(name))
    return fnames


def prepareData(dir):
    ''' Input:
            text file, each lines is a sample,
            the last token in the line is the class label.
        Output:
            sampleTexts = list of strings, each string is a sample;
            y = list of class labels.
    '''
    fnames = list_files(dir)

    sampleTexts=[]
    y=[]
    exclude = set(string.punctuation)
    exclude.update(['…','¿', '¡', '``'])

    for fname in fnames:
        f=open(dir+'/'+fname,encoding = "latin-1")
        text=f.read()
        text=text.replace('\n', ' ')
        text=text.lower()

        words=nltk.word_tokenize(text.strip())
        wordsClean=[]

        for word in words:
            word = ''.join(ch for ch in word if ch not in exclude)
            if word != '':
                wordsClean.append(word)

        text=' '.join(wordsClean)
        sampleTexts.append(text)

        f.close()
        if 'no' in fname:
            y.append(0)
        elif 'yes' in fname:
            y.append(1)

    return sampleTexts, y

def getPOL():

    f = open('../ML-SentiCon/senticon.es.xml',encoding = "latin-1")
    xml = f.read()
    f.close()
    tabla = {}
    soup = BeautifulSoup(xml, 'lxml')
    lemas=soup.findAll('lemma')

    for lema in lemas:
        text = lema.get_text()
        text = text.replace(' ', '')
        pol = lema.attrs['pol']
        tabla[text] = float(pol)

    return tabla

def prepareData_xml(dir):
    ''' Input:
            xml file with reviews ranked from 1 to 5.
        Output:
            sampleTexts = list of strings, each string is a sample;
            y = list of class labels.
    '''
    fnames = list_files_xml(dir)

    sampleTexts=[]
    y=[]
    exclude = set(string.punctuation)
    exclude.update(['…','¿', '¡', '``'])

    for fname in fnames:
        f=open(dir+'/'+fname,encoding = "latin-1")
        xml=f.read(); f.close()

        soup = BeautifulSoup(xml, 'lxml')
        body=soup.find('body')
        text=body.get_text()

        reviewTag=soup.find('review')
        rank=reviewTag.attrs['rank']

        text=text.replace('\n', ' ')
        text=text.lower()

        words=nltk.word_tokenize(text.strip())
        wordsClean=[]

        for word in words:
            word = ''.join(ch for ch in word if ch not in exclude)
            if word != '':
                wordsClean.append(word)

        text=' '.join(wordsClean)
        sampleTexts.append(text)
        y.append(int(rank))

    return sampleTexts, y

def prepareData_lemmas(dir):
    ''' Input:
            xml file of lemmas with reviews ranked from 1 to 5.
        Output:
            sampleTexts = list of strings, each string is a sample;
            y = list of class labels.
    '''
    fnames_pos = sorted(list_files_pos(dir))
    fnames_xml = sorted(list_files_xml(dir))
    tabla = getPOL()
    sumas = []
    for fname in fnames_pos:
        suma = 0
        f = open(dir+'/'+fname, encoding = "latin-1")
        lines = f.readlines()
        f.close()
        exclude = set(string.punctuation)
        exclude.update(['…','¿', '¡', '``'])
        lemmas=[]
        for line in lines:
            if line != '':
                line=line.strip()
                words=nltk.word_tokenize(line.lower())
                if words:
                    if words[0] not in exclude:
                        if words[1] in tabla:
                            suma = suma + tabla[words[1]]
        sumas.append(suma)
    y = []
    for fname in fnames_xml:
        f=open(dir+'/'+fname,encoding = "latin-1")
        xml=f.read(); f.close()

        soup = BeautifulSoup(xml, 'lxml')
        reviewTag=soup.find('review')
        rank=reviewTag.attrs['rank']
        y.append(int(rank))

    sumasT = [0,0,0,0,0]
    cantidad = [0,0,0,0,0]

    for i in range(len(sumas)):
        if i>=len(y):
            print('Ya se hubiera muerto ', i)
            i = i % len(y)
        print(y[i],"       ",sumas[i%len(sumas)])
        sumasT[y[i]-1] = sumasT[y[i]-1] + sumas[i];
        cantidad[y[i]-1] = cantidad[y[i]-1] + 1;

    total = [0,0,0,0,0]
    for i in range(len(sumasT)):
         total[i] = sumasT[i] / cantidad[i]
    return total

if __name__=='__main__':

    dir='../corpusCriticasCine'

    sampleTexts, y = prepareData_lemmas(dir)

    print(len(sampleTexts))
    print(len(y))
