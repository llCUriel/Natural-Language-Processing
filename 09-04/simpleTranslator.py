from nltk.corpus import swadesh
from nltk.corpus import wordnet as wn

# print(swadesh.fileids())

es2en = swadesh.entries(['es', 'en'])

translate = dict(es2en)

words = ['cenizas', 'nieve', 'hincharse']

for w in words:
    print(w, " = ", translate[w])

# for w in translate:
    # print(w)

computer = wn.synsets('computer')[0]
print("The hypernyms of computer are ", computer.hypernyms())
print("The hyponyms of computer are ", computer.hyponyms())

automobile = wn.synsets('automobile')[0]
print("The meronyms of automobile are ", automobile.part_meronyms())

bird = wn.synsets('bird')[0]
print("The holonyms of bird are ", bird.member_holonyms())
