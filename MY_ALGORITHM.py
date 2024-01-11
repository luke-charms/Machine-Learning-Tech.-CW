import pandas
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from langdetect import detect
from googletrans import Translator

import matplotlib.pyplot as plt


#_____________________________________
#======= Data Characterization =======
#-------------------------------------

trainingData = pandas.read_csv("dataset/mediaeval-2015-trainingset.txt", sep="\t")
testingData = pandas.read_csv("dataset/mediaeval-2015-testset.txt", sep="\t")

# create pandas dataframes for training and testing
training_df = pandas.DataFrame(data = trainingData)
testing_df = pandas.DataFrame(data = testingData)

# size of training and testing dataset
training_df.shape
testing_df.shape

# general information about both datasets
trainingData.info()
testingData.info()

# rename certain columns and find number of tweets about event for training
training_df.rename(columns = {'imageId(s)':'Images'}, inplace = True)
imgCount = training_df.groupby(training_df.Images.str.split('_').str[0])['tweetId'].nunique()
print (imgCount)

# plot event frequency on bar chart
#imgCount.plot(kind='bar')
#plt.tight_layout()
#plt.figure(1)

# rename certain columns and find number of tweets about event for testing
testing_df.rename(columns = {'imageId(s)':'Images'}, inplace = True)
imgCount = testing_df.groupby(testing_df.Images.str.split('_').str[0])['tweetId'].nunique()
print (imgCount)

# plot event frequency on bar chart
#imgCount.plot(kind='bar')
#plt.tight_layout()
#plt.figure(2)


# find all different languages of the tweetTexts
'''
langs = dict()

for tweetText in training_df['tweetText']:
    try:
        language = detect(tweetText)
    except:
        pass
        language = "N/A"

    if language in langs.keys():
        langs[language] = langs[language] + 1
    else:
        langs[language] = 1


print ("Number of different languages: ", langs)
'''


#__________________________________
#======= Data Preprocessing =======
#----------------------------------


#Changing 'humor' to 'fake'
training_df.loc[(training_df.label == 'humor'), 'label'] = 'fake'
testing_df.loc[(testing_df.label == 'humor'), 'label'] = 'fake'



# function to translate tweetText
def translate_tweet(tweetText):
    translator = Translator()
    translator.raise_Exception = True
    try:
        translation = translator.translate(tweetText, dest='en').text
        return translation
    except:
        print("unable to translate tweet: ", tweetText)
        return tweetText


# remove all unnecessary non-textual features and noise from tweetText
def remove_noise(tweetText):
    # removes URLs
    tweetText = re.sub(r'http\S+','', tweetText)
    tweetText = re.sub(r'\\\/\S+', '', tweetText)
    # removes usernames
    tweetText = re.sub(r'@\w+', "", tweetText)
    # removes '&'s and newline symbol
    tweetText = re.sub(r'&amp;', '', tweetText)
    tweetText = re.sub(r'\\n', '', tweetText)
    # removes non-word characters
    tweetText = re.sub(r'[^\w\s]','',tweetText)
    # removes numbers
    tweetText = re.sub(r'[0-9]','',tweetText)

    emoticons = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags = re.UNICODE)
    # removes emoticons
    tweetText = emoticons.sub(r'', tweetText)

    # translate non-English tweets
    #if detect(tweetText) != 'en':
    #    tweetText = translate_tweet(tweetText)
    
    return tweetText




# remove text noise using function above
training_df['cleanedText'] = training_df['tweetText'].apply(lambda text: remove_noise(text))
testing_df['cleanedText'] = testing_df['tweetText'].apply(lambda text: remove_noise(text))
training_df.head()


# initiate stopwords remover and extend with punctuation
stopwords = nltk.corpus.stopwords.words()
#stopwords.extend([':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', '...'])

# removes stopwords
training_df['cleanedText'] = training_df['cleanedText'].apply(lambda text: ' '.join([x for x in text.split() if x not in stopwords]))
testing_df['cleanedText'] = testing_df['cleanedText'].apply(lambda text: ' '.join([x for x in text.split() if x not in stopwords]))


#Removing whitespace
training_df['cleanedText'] = training_df['cleanedText'].apply(lambda text: " ".join(text.split()))
testing_df['cleanedText'] = testing_df['cleanedText'].apply(lambda text: " ".join(text.split()))


# initiate lemmatiser
tokeniser = nltk.tokenize.WhitespaceTokenizer()
lemmatiser = nltk.stem.WordNetLemmatizer()

# lemmatise tweetText
training_df['cleanedText'] = training_df['cleanedText'].apply(lambda text: ' '.join([lemmatiser.lemmatize(x) for x in tokeniser.tokenize(text)]))
testing_df['cleanedText'] = testing_df['cleanedText'].apply(lambda text: ' '.join([lemmatiser.lemmatize(x) for x in tokeniser.tokenize(text)]))
training_df.head()


# remove duplicate rows
training_df.drop_duplicates(subset=['cleanedText'], keep='first', inplace=True)



#_____________________________________________
#======= Algorithm Design and Training =======
#---------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics



# define targets and features for training and testing dataset
training_targets = training_df.label
training_features = training_df.cleanedText
testing_targets = testing_df.label
testing_features = testing_df.cleanedText


# initialise TF-IDF
tfidf_vectoriser =  TfidfVectorizer(stop_words='english', max_df=0.05)
tfidf_train = tfidf_vectoriser.fit_transform(training_features)
tfidf_test = tfidf_vectoriser.transform(testing_features)


# initialise MNB classifier
multinomailNB_clf = MultinomialNB()
multinomailNB_clf.fit(tfidf_train, training_targets)
multinomailNB_pred = multinomailNB_clf.predict(tfidf_test)

# initialise LG classifier
logisticReg_clf = LogisticRegressionCV(cv=20, max_iter=7500)
logisticReg_clf.fit(tfidf_train, training_targets)
logisticReg_pred = logisticReg_clf.predict(tfidf_test)

"""
logisticReg_clf2 = LogisticRegressionCV(cv=30, max_iter=7500)
logisticReg_clf2.fit(tfidf_train, training_targets)
logisticReg_pred2 = logisticReg_clf2.predict(tfidf_test)

logisticReg_clf3 = LogisticRegressionCV(cv=40, max_iter=7500)
logisticReg_clf3.fit(tfidf_train, training_targets)
logisticReg_pred3 = logisticReg_clf3.predict(tfidf_test)
"""


# Function to calculate F1 score and print results

def calculate_f1Score(model_pred):
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for model_pred, test_target in zip(model_pred, testing_targets):
        if model_pred == 'fake':
            if test_target == 'fake':
                TP += 1
            else:
                FP += 1
        elif model_pred == 'real':
            if test_target == 'fake':
                FN += 1
            else:
                TN +=1
            
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    return f1_score




# print metrics and accuracy for MNB
print("MNB accuracy:   %0.3f" % metrics.accuracy_score(testing_targets, multinomailNB_pred))
print("MNB F1 Score is: ", metrics.f1_score(testing_targets, multinomailNB_pred, pos_label="real"))
print("MNB F1 score (calculated) is: ", calculate_f1Score(multinomailNB_pred))

confusion_matrix_MNB = metrics.confusion_matrix(testing_targets, multinomailNB_pred, labels=multinomailNB_clf.classes_)
cm_display_MNB = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_MNB,  display_labels=multinomailNB_clf.classes_)
cm_display_MNB.plot()


# print metrics and accuracy for LG
print("LG accuracy:   %0.3f" % metrics.accuracy_score(testing_targets, logisticReg_pred))
print("LG F1 Score is: ", metrics.f1_score(testing_targets, logisticReg_pred, pos_label="real"))
print("LG F1 score (calculated) is: ", calculate_f1Score(logisticReg_pred))

confusion_matrix_LG = metrics.confusion_matrix(testing_targets, logisticReg_pred, labels=logisticReg_clf.classes_)
cm_display_LG = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_LG,  display_labels=logisticReg_clf.classes_)
cm_display_LG.plot()

plt.show()