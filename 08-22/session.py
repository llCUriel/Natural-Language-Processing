# coding: utf-8
import nltk
from nltk.book import *
text5.concordance("great")
text5.concordance("day")
text5.count("day")
text5.count('day')
text5.concordance('day')
text5.concordance('days')
text5.concordance(['days', 'days'])
text5.concordance('days')
text7.concordance('arms')
text1.concordance('terrible')
text1.similar('terrible')
text2.similar('terrible')
text5.similar('terrible')
text6.similar('terrible')
text6.similar('sad')
text6.similar('terrible')
text6.similar(['terrible', 'sad'])
text6.similar('sad')
text6.common_contexts(['sad', 'terrible'])
text5.common_contexts(['sad', 'terrible'])
text7.dispersion_plot(['bank', 'day', 'great', 'president', 'meeting', 'money', 'debt', 'speal'])
text7.dispersion_plot(['bank', 'day', 'great', 'president', 'meeting', 'money', 'debt', 'speak'])
text7.dispersion_plot(['bank', 'day', 'great', 'president', 'meeting', 'money', 'debt', 'speak'])
len(text8)
len(set(text8))
type(set(text8))
len(text8)/len(set(text8))
[print (text8.count(w)) for w in set(text8)]
print (text8.count(w)) for w in set(text8)
ws = [text8.count(w) for w in set(text8)]
ws
ws = [(text8.count(w), w) for w in set(text8)]
ws
text1.count("terrible")/len(set(text1))
100*text1.count("terrible")/len(set(text1))
100*text1.count("terrible")/len(text1)
$save books.py
$save books
$save today  ~0/ 
get_ipython().set_next_input('$save');get_ipython().run_line_magic('pinfo', 'save')
$save session 1-44
