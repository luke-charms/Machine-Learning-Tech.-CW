import pandas
import numpy
import re
import nltk

from langdetect import detect
from googletrans import Translator

# Setting training and testing datasets
trainingData = pandas.read_csv("dataset/mediaeval-2015-trainingset.txt", sep="	")
testingData = pandas.read_csv("dataset/mediaeval-2015-testset.txt", sep="	")

#Creating DataFrames for training and testing
df_train = pandas.DataFrame(data = trainingData)
df_test = pandas.DataFrame(data = testingData)


#_____________________________________
#======= Data Characterization =======
#-------------------------------------

trainingData.head()

trainingData.info() #Metadata of training data, including size

df_test.shape #Size of testing data


#Determine events covered and their frequency by image names in training data
df_train.rename(columns = {'imageId(s)':'imgs'}, inplace = True)
imgCount = df_train.groupby(df_train.imgs.str.split('_').str[0])['tweetId'].nunique()
print(imgCount)

#Determine events covered and their frequency by image names in testing data
df_test.rename(columns = {'imageId(s)':'imgs'}, inplace = True)
imgCount = df_test.groupby(df_test.imgs.str.split('_').str[0])['tweetId'].nunique()
print (imgCount)


#Helper to look into the tweetText of a particular event image to determine what the event is
selector = []
for imgs in df_train['imgs']:
    if "sandy" in imgs:
        selector.append(True)
    else:
        selector.append(False)
        
isEvent = pandas.Series(selector)

df_event = df_train[isEvent].head(61)

for tweet in df_event['tweetText']:
    print(tweet)


langs = dict()

for tweet in df_train['tweetText']:
    try:
        lan = detect(tweet)
    except:
        pass
        lan = "Unknown"
        print(tweet)
    if lan in langs.keys():
        langs[lan] = langs[lan] + 1
    else:
        langs[lan] = 1


print (langs)

#__________________________________
#======= Data Preprocessing =======
#----------------------------------

#Changing 'humor' to 'fake'
df_train.loc[(df_train.label == 'humor'),'label'] = 'fake'
df_test.loc[(df_test.label == 'humor'),'label'] = 'fake'


#Removing retweets, reposts, and modified tweets
rtPattern1 = "(RT|rt|MT|mt|RP|rp):? @\w*:?"
rtPattern2 = "(\bRT\b|\brt\b|\bMT\b|\bmt\b|\bRP\b|\brp\b)"
rtPattern3 = "(@\w*:)"
rtPattern4 = "(#rt|#RT|#mt|#MT|#rp|#retweet|#Retweet|#modifiedtweet|#modifiedTweet|#ModifiedTweet|#repost|#Repost)"
rtPattern5 = "(via @\w*)"

retweets = df_train['tweetText'].str.contains(rtPattern1)
df_train = df_train[~retweets]

retweets = df_train['tweetText'].str.contains(rtPattern2)
df_train = df_train[~retweets]

retweets = df_train['tweetText'].str.contains(rtPattern3)
df_train = df_train[~retweets]

retweets = df_train['tweetText'].str.contains(rtPattern4)
df_train = df_train[~retweets]

retweets = df_train['tweetText'].str.contains(rtPattern5)
df_train = df_train[~retweets]

df_train.reset_index(drop=True, inplace=True)
df_train.shape


#Removing remaining twitter handles @username
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'@\w*', "", text))


#Removing emojis
emojis = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)

df_train['tweetText'] = df_train['tweetText'].apply(lambda text: emojis.sub(r'', text) if emojis.search(text) else text)


#Cleaning symbols - ampersand and newline
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'&amp;|\\n', '', text))


#Removing urls
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'http\S+', '', text))
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'\\\/\S+', '', text))


'''
#NOT USED
#Translating to English

i = 0
for tweet in df_train['tweetText']:
    try:
        lan = detect(tweet)
    except:
        continue
    if lan != 'en':
        tr = Translator()
        trText = tr.translate(tweet).text
        print(trText)
        #TODO replace cell value to translation
    i = i + 1
'''

#Removing whitespace
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: " ".join(text.split()))


#Initialise stopwords
stopwords = nltk.corpus.stopwords.words()
stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...'])


#Removing stopwords
df_train['filteredTweet'] = df_train['tweetText'].apply(lambda x: ' '.join([w for w in x.split() if w not in stopwords]))


#Lemmatising
tokeniser = nltk.tokenize.WhitespaceTokenizer()
lemmatiser = nltk.stem.WordNetLemmatizer()

df_train['lemmatisedTweet'] = df_train['filteredTweet'].apply(lambda x: ' '.join([lemmatiser.lemmatize(w) for w in tokeniser.tokenize(x)]))
df_train.head(10)



#_____________________________________________
#======= Algorithm Design and Training =======
#---------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import metrics


#Define features and target for training and testing
tar_train = df_train.label
ft_train = df_train.lemmatisedTweet
tar_test = df_test.label
ft_test = df_test.tweetText


#Init Bag-of-Words
count_vectoriser = CountVectorizer(stop_words='english')
count_train = count_vectoriser.fit_transform(ft_train)
count_test = count_vectoriser.transform(ft_test)


#Init N-Gram
ngram_vectoriser = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
ngram_train = ngram_vectoriser.fit_transform(ft_train)
ngram_test = ngram_vectoriser.transform(ft_test)


#Init TF-IDF
tfidf_vectoriser = TfidfVectorizer(stop_words='english', max_df=0.2)
tfidf_train = tfidf_vectoriser.fit_transform(ft_train)
tfidf_test = tfidf_vectoriser.transform(ft_test)


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#======= CHOOSE TWO TO IMPLEMENT =======
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Multinomial Naive Bayes Classification
clf = MultinomialNB()

# Bernoulli Naive Bayes Classification
clf = BernoulliNB()

# Passive Aggressive Classification
clf = PassiveAggressiveClassifier()

# Stochastic Gradient Descent Classification
clf = SGDClassifier()


#Bag-of-Words
clf.fit(count_train, tar_train)

pred = clf.predict(count_test)
score = metrics.accuracy_score(tar_test, pred)

print("accuracy:   %0.3f" % score)


#N-Grams
clf.fit(ngram_train, tar_train)

pred = clf.predict(ngram_test)
score = metrics.accuracy_score(tar_test, pred)

print("accuracy:   %0.3f" % score)


#TF-IDF
clf.fit(tfidf_train, tar_train)

pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(tar_test, pred)

print("accuracy:   %0.3f" % score)

f1Score = metrics.f1_score(tar_test, pred, average='macro')

print("F1 Score is: ", f1Score)