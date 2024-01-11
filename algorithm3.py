import urllib
from io import BytesIO
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from langdetect import detect

from nltk.tokenize import WhitespaceTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

np.random.seed(69)


# Import the data

df_training = pd.read_csv(r"dataset/mediaeval-2015-trainingset.txt", sep="\t")
df_testing = pd.read_csv(r"dataset/mediaeval-2015-testset.txt", sep="\t")





# Change the humour label to the fake label
df_training.loc[df_training.label=='humor', 'label'] = 'fake'
df_testing.loc[df_testing.label=='humor', 'label'] = 'fake'


# Function for removing substrings from tweets
def remove_pattern(tweet_text, pattern):
    return re.sub(pattern, '', tweet_text)


# Remove URLs
url_pattern1 = r'http\S+'
url_pattern2 = r'\\\/\S+'

def remove_url1(tweet_text):
    return remove_pattern(tweet_text, url_pattern1)

def remove_url2(tweet_text):
    return remove_pattern(tweet_text, url_pattern2)

df_training['tweetText'] = df_training['tweetText'].apply(remove_url1)
df_training['tweetText'] = df_training['tweetText'].apply(remove_url2)



# Remove @ mentions
username_pattern = r'@\w+'
def remove_username(tweet_text):
    return remove_pattern(tweet_text, username_pattern)

df_training['tweetText'] = df_training['tweetText'].apply(remove_username)


# Remove symbols
newline_pattern = r'&amp;|\\n'
def remove_newline(tweet_text):
    return remove_pattern(tweet_text, newline_pattern)

df_training['tweetText'] = df_training['tweetText'].apply(remove_newline)


# Remove emojis
def remove_emojis(tweet_text):
    emoji_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"    # Unicode values for emojis
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags = re.UNICODE)
    return emoji_pattern.sub(r'', tweet_text)

df_training['tweetText'] = df_training['tweetText'].apply(remove_emojis)


# Remove stopwords
stopword_list = stopwords.words()

def remove_stopwords(tweet_text):
    words = []
    for word in tweet_text.split():
        if word not in stopword_list:
            words.append(word)
    return ' '.join(words)

df_training['importantText'] = df_training['tweetText'].apply(remove_stopwords)


# Tokenise and lemmatise the tweet text
tokeniser = WhitespaceTokenizer()
lemmatiser = WordNetLemmatizer()

def lemmatise_text(tweet_text):
    tokenised_text = tokeniser.tokenize(tweet_text)
    lemmatised_text = [lemmatiser.lemmatize(word) for word in tokenised_text]
    return ' '.join(lemmatised_text)

df_training['lemmatisedText'] = df_training['importantText'].apply(lemmatise_text)



# Set up training predictor and target variables

training_predictor = df_training['lemmatisedText']
training_target = df_training['label']
# Set up testing predictor and target variables

testing_predictor = df_testing['tweetText']
testing_target = df_testing['label']


# Vectorise the words with TF-IDF

tfidf_vectoriser = TfidfVectorizer(stop_words='english', max_df=0.1)
tfidf_vectoriser.fit(df_training['lemmatisedText'])

training_tfidf = tfidf_vectoriser.transform(training_predictor)
testing_tfidf = tfidf_vectoriser.transform(testing_predictor)


# Train multinomial Naive-Bayes model with TF-IDF
multi_nb = naive_bayes.MultinomialNB()
multi_nb.fit(training_tfidf, training_target)

# Predict labels for testing data
multi_nb_pred = multi_nb.predict(testing_tfidf)


# Function to calculate F1 score and print results

def f1_score(prediction):
    TP, FP, TN, FN = 0, 0, 0, 0
    
    for prediction, target in zip(prediction, testing_target):
        if prediction == 'fake':
            if target == 'fake':
                TP += 1
            elif target == 'real':
                FP += 1
        elif prediction == 'real':
            if target == 'fake':
                FN += 1
            elif target == 'real':
                TN +=1
            
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    print('TP: ' + str(TP) + ' FP: ' + str(FP) + ' TN: ' + str(TN) + ' FN: ' + str(FN))
    print('F1 score: ' + str(f1))
    
    return [f1, TP, FP, TN, FN]


print("Multinomial Naive Bayes accuracy score : ", accuracy_score(multi_nb_pred, testing_target) * 100)
multi_nb_tfidf_R = f1_score(multi_nb_pred)