import pandas
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from googletrans import Translator

import matplotlib.pyplot as plt


trainData = pandas.read_csv("dataset/mediaeval-2015-trainingset.txt", sep="\t")
testData = pandas.read_csv("dataset/mediaeval-2015-testset.txt", sep="\t")

#Creating DataFrames for training and testing
df_train = pandas.DataFrame(data = trainData)
df_test = pandas.DataFrame(data = testData)


trainData.head()

testData.head()

df_train.shape

df_test.shape

trainData.describe(include=object)

testData.describe(include=object)


trainData.info()

testData.info()

# Checking if any null value in training data
df_train.isnull().sum().sum()


# Checking if any null value in testing data
df_test.isnull().sum().sum()

df_train.rename(columns = {'imageId(s)':'Images'}, inplace = True)
imgCount = df_train.groupby(df_train.Images.str.split('_').str[0])['tweetId'].nunique()
print (imgCount)

# Plot a bar chart of the frequencies
imgCount.plot(kind='bar')
plt.tight_layout()
plt.figure(1)

df_test.rename(columns = {'imageId(s)':'Images'}, inplace = True)
imgCount = df_test.groupby(df_test.Images.str.split('_').str[0])['tweetId'].nunique()
print (imgCount)

# Plot a bar chart of the frequencies
imgCount.plot(kind='bar')
plt.tight_layout()
plt.figure(2)

#plt.show()



#----------------------------------------------------------
#============ DATA PREPROCESSING ==========================
#----------------------------------------------------------


df_train.loc[(df_train.label == 'humor'),'label'] = 'fake'
df_test.loc[(df_test.label == 'humor'),'label'] = 'fake'


# Removing URLs
def removeURLs(text):
    tweet = re.sub(r'http\S+','', text)
    return tweet

df_train['tweetText']= df_train['tweetText'].apply(lambda x: removeURLs(x))
df_test['tweetText'] = df_test['tweetText'].apply(lambda x: removeURLs(x))

pandas.set_option('display.max_colwidth', None) #allows you to see full fields
df_train['tweetText'].head(10)

# Translating tweets
def translate_tweet(text):
    translator = Translator()
    translator.raise_Exception = True
    try:
        translation = translator.translate(text, dest='en').text
        return translation
    except:
        return text
    

#remove non word characters, &amp, new line indicators, convert to same case and translate
def clean_text(tweet):
    tweet = re.sub(r'&\S+', '', tweet) # remove '&amp'
    tweet = tweet.replace("\\n",'') # remove end of line signs '\n'
    tweet = re.sub(r'[^\w\s]','',tweet) # remove non word characters
    tweet = re.sub(r'@\w*', "", tweet) # remove usernames
    tweet = tweet.lower() #convert to lower case
    tweet = re.sub(r'[0-9]','',tweet) #remove numbers
    emojis = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
    tweet = emojis.sub(r'', tweet) if emojis.search(tweet) else tweet # remove emojis
    
    # if lang_detect(tweet) != 'en':
    #     tweet = translate_tweet(tweet) #translate to english
    return tweet


df_train['cleanText'] = df_train['tweetText'].apply(lambda x: clean_text(x))
df_test['cleanText'] = df_test['tweetText'].apply(lambda x: clean_text(x))
df_train.head()

stopwords = nltk.corpus.stopwords.words()
stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...'])
df_train['cleanText'] = df_train['cleanText'].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))
df_test['cleanText'] = df_test['cleanText'].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))


tokeniser = nltk.tokenize.WhitespaceTokenizer()
lemmatiser = nltk.stem.WordNetLemmatizer()
df_train['cleanText'] = df_train['cleanText'].apply(lambda x: ' '.join([lemmatiser.lemmatize(w) for w in tokeniser.tokenize(x)]))
df_test['cleanText'] = df_test['cleanText'].apply(lambda x: ' '.join([lemmatiser.lemmatize(w) for w in tokeniser.tokenize(x)]))


df_train['frequency'] = df_train['tweetText'].map(df_train['tweetText'].value_counts())
df_train['frequency'].value_counts()
df_train.drop_duplicates(subset=['tweetText'], keep='first', inplace=True)
df_train.shape


df_train['cleanText'] = df_train['cleanText'].apply(lambda text: " ".join(text.split()))
df_train.head(10)



#----------------------------------------------------------
#============ Training Algorithms =========================
#----------------------------------------------------------


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import seaborn as sn


tar_train = df_train.label
ft_train = df_train.cleanText
tar_test = df_test.label
ft_test = df_test.cleanText


tfidf_vectoriser = TfidfVectorizer(stop_words='english', max_df=0.2)
tfidf_train = tfidf_vectoriser.fit_transform(ft_train)
tfidf_test = tfidf_vectoriser.transform(ft_test)


multinomailNB_clf = MultinomialNB()
multinomailNB_clf.fit(tfidf_train, tar_train)
multinomailNB_pred = multinomailNB_clf.predict(tfidf_test)

score = metrics.accuracy_score(tar_test, multinomailNB_pred)
f1_score = metrics.f1_score(tar_test, multinomailNB_pred, pos_label="real")

print("accuracy:   %0.3f" % score)
print("F1 Score:", f1_score)

lg_clf = LogisticRegressionCV(cv=5, random_state=0, dual=False, max_iter=7600)
lg_clf.fit(tfidf_train, tar_train)
lg_pred = lg_clf.predict(tfidf_test)

score = metrics.accuracy_score(tar_test, lg_pred)
f1_score = metrics.f1_score(tar_test, lg_pred, pos_label="real")

print("accuracy:   %0.3f" % score)
print("F1 Score:", f1_score)

