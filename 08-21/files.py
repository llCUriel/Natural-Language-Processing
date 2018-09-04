# Reading the e960401 file for basic manipulation

import locale

# This library is for a HTML Parser
from bs4 import BeautifulSoup

print(locale.getdefaultlocale())

text = ""
# Read the hole file
with open("../../Corpus/e960401.htm", encoding='latin-1') as f:
# The test has a sample of the hole file, only one paragraph
# with open("test.htm", encoding='latin-1') as f:
    print(f)
    text = f.read()
    print(len(text))
    # print(text)

soup = BeautifulSoup(text, 'lxml')
parsedText = soup.get_text()
parsedText = parsedText.replace('\x97',' ')

print(parsedText)
print(len(parsedText))

with open("test.txt", "w") as f:
    f.write(parsedText)


