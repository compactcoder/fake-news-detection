# Importing libraries for cleaning puprose
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Data cleaning function
def clean_words(text):

    cleantext = ''
    text = text.lower() # Converting all text data into lowercase

    # Simplifying text
    text = re.sub(r'i\'m', 'i am', text)
    text = re.sub(r'he\'s', 'he is', text)
    text = re.sub(r'she\'s', 'she is', text)
    text = re.sub(r'that\'s', 'that is', text)
    text = re.sub(r'what\'s', 'what is', text)
    text = re.sub(r'where\'s', "where is", text)
    text = re.sub(r'\'ll', " will", text)
    text = re.sub(r'\'ve', " have", text)
    text = re.sub(r'\'re', ' are', text)
    text = re.sub(r'\'d', ' would', text)
    text = re.sub(r'won\'t', 'will not', text)
    text = re.sub(r'can\'t', 'cannot', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)  # removes links eg. http://url.com/bla1
    # text = re.sub('\[.*?\]','',text)
    text = re.sub(r'\W', ' ', text)  # removes non word character
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # removes punctuation marks
    text = re.sub(r'\n', '', text)  # removes new line character
    text = re.sub('\w*\d\w*', '', text)  # removes word wich contain number eg. 12is, trum5
    text = re.sub(r'\s+', ' ', text)

    Lem = WordNetLemmatizer()  # intialiting Lemmtizer object
    stopwordslist = stopwords.words('english')  # Stopwords of NLTK

    # Removing stopwords as well as lemmetizing words
    for word in text.split():
        if word not in stopwordslist:
            cleantext += Lem.lemmatize(word) + ' '

    return cleantext
