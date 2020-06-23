from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import re
from nltk import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns

def get_wordnet_pos(wt):
    (word, tag) = wt
    tb_tag = wordnet.NOUN
    
    if tag.startswith('J'):
        tb_tag = wordnet.ADJ
    elif tag.startswith('V'):
        tb_tag = wordnet.VERB
    elif tag.startswith('N'):
        tb_tag = wordnet.NOUN
    elif tag.startswith('R'):
        tb_tag = wordnet.ADV
    
    return (word, tb_tag)

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won(\'|’)t", "will not", phrase)
    phrase = re.sub(r"can(\'|’)t", "can not", phrase)
    phrase = re.sub(r"idk", "i do not know", phrase)

    # typos
    phrase = re.sub(r" arent ", " are not ", phrase)
    phrase = re.sub(r" doesnt ", " does not ", phrase)
    phrase = re.sub(r" dont ", " do not ", phrase)

    # general
    phrase = re.sub(r"n(\'|’)t", " not", phrase)
    phrase = re.sub(r"(\'|’)re", " are", phrase)
    phrase = re.sub(r"(\'|’)s", " is", phrase)
    phrase = re.sub(r"(\'|’)d", " would", phrase)
    phrase = re.sub(r"(\'|’)ll", " will", phrase)
    phrase = re.sub(r"(\'|’)t", " not", phrase)
    phrase = re.sub(r"(\'|’)ve", " have", phrase)
    phrase = re.sub(r"(\'|’)m", " am", phrase)
    return phrase

def lemmatization_spacy(nlp, sent): # filter noun and adjective
    doc = nlp(sent) 
    
    return [token.lemma_ for token in doc if not token.is_stop]

def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()
    
def replace_numbers(review):
    return ' '.join(re.sub("[0-9]+", "NUMBER", review).split())

def only_letters(review):
    return ' '.join(re.sub("[^a-zA-Z]", " ", review).split())

# Lematiza una oración
def lemmatize(lemmatizer, r):
    return [lemmatizer.lemmatize(w, t) for (w, t) in r]
    
# Remueve palabras de menos de 3 caracteres
def discard_small_words(r):
    return ' '.join(list(filter(lambda x: len(x) > 2, r.split())))

# Remueve de una review las palabras que estén en una wordlist
def remove_wordlist(review, wordlist):
    return ' '.join(list(filter((lambda x: x not in wordlist), review.split())))

def reduce_lengthening(text):
    pattern = re.compile("(.)\\1{2,}")
    return pattern.sub("\\1\\1", text)
    
