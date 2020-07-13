'''
This program is built to test the basic sentiment analyzer.

The purpose for this program is to gauge the efficiency of using the NLTK
library


FYI:    Using " #%% " is for the purpose of debugging using jupyter notebook  

'''

#%%
import os
import nltk
import matplotlib.pyplot as plt
import random
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode




# Split the data into two sets, positive and negative

files_pos = os.listdir('train/pos')
files_pos = [open('train/pos/' + f, 'r').read() for f in files_pos]
files_neg = os.listdir('train/neg')
files_neg = [open('train/neg/' + f, 'r').read() for f in files_neg]

all_words = []
documents = []

stop_words = list(set(stopwords.words('english')))

#  j is adject, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for p in files_pos:
    documents.append((p, "pos"))

    # remove punctuations-Regex Expression
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # tokenize
    tokenized = word_tokenize(cleaned)
    
    # remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # parts of speech tagging for each word
    pos = nltk.pos_tag(stopped)

    # make a list of  all adjectives identified by the allowed word types list above
for w in pos:
    if w[1][0] in allowed_word_types:
        all_words.append(w[0].lower())

for p in files_neg:
    # create a list of tuples where the first element of each tuple is a review
    # the second element is the label
    documents.append((p, "neg"))

    # remove punctuations
    cleaned = re.sub(r'[^(a-zA-Z)\s]', '', p)

    # tokenize
    tokenized = word_tokenize(cleaned)

    # remove stopwords
    stopped = [w for w in tokenized if not w in stop_words]

    # parts of speech tagging for each word
    neg = nltk.pos_tag(stopped)

    # make a list of  all adjectives identified by the allowed word types list above
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

'''
    Create features for each individual Review

    #Tuple feature set for a given review
    ({'great': True, 'excellent': False, 'horrible': False}, 'pos') 



    Split the Feauture_set to training set (20, 000) and testing set (5,000)

'''

# creating a frequency distribution of each adjectives.
all_words = nltk.FreqDist(all_words)

# listing the 5000 most frequent words
word_features = list(all_words.keys())[:5000]


# function to create a dictionary of features for each review in the list document.
# The keys are the words in word_features
# The values of each key are either true or false for weather that feature appears in the review or not

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


    # Creating features for each review
feature_sets = [(find_features(rev), category) for (rev, category) in documents]

# Shuffling the documents
random.shuffle(feature_sets)

training_set = feature_sets[:20000]
testing_set = feature_sets[20000:]


'''

    :: Naive Bayes Classifier

    Naive Bayes is a classification algorithm that is a part of the NLTK library. 
    for binary (two-class) and multi-class classification problems. 

    https://machinelearningmastery.com/naive-bayes-for-machine-learning/

'''





classifier = nltk.NaiveBayesClassifier.train(training_set)
print(" Classifier accuracy percentage:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(25)





'''

    The list to above (left) shows 15 of the most informative features from the model. 
    And the ratios associated with them shows how much more often each corresponding word appear 
    in one class of text over others. These ratios are known as likelihood ratios.

    evaluate the model I calculated the f1_score using sci-kit learn and created a confusion matrix. 
    The f1_score was 84.36%. The normalized confusion matrix shows that the model predicted correctly 
    for 83% of the positive reviews and 85% of the negative reviews.

'''


#from sklearn.metrics import f1_score
#f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')




'''

    NLTK has a builtin Scikit Learn module called SklearnClassifier. This SklearnClassifer can inherit 
    the properties of any model that you can import through Scikit Learn

    -> NLTK SkleanClassifier

    To see/test other classifiers use this link: 
    https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn


    With this classifier we will be implementing the following classifiers: 

    Multinomial Naive Bayes      -   MNB_clf
    Bernoulli Naive Bayes        -   BNB_clf
    Logistic Regression          -   LR_clf
    Stochastic Gradient Descent  -   SGD_clf
    Support Vector Classifier    -   SV_clf

    The purpose of implementing different classifier algorthims allows us find a model 
    with the best score


'''


MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)

LR_clf = SklearnClassifier(LogisticRegression())
LR_clf.train(training_set)

SGD_clf = SklearnClassifier(LogisticRegression())
SGD_clf.train(training_set)

SV_clf = SklearnClassifier(SVC())
SV_clf.train(training_set)


from sklearn.metrics import f1_score, accuracy_score
ground_truth = [r[1] for r in testing_set]
predictions = {}
f1_scores = {}

for clf, listy in classifier.items(): 
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    f1_scores[clf] = f1_score(ground_truth, predictions[clf])
    print(f'f1_score {clf}: {f1_scores[clf]}')

print(" Accuracy: MNB: 0.845, BNB: 0.8447999, LogReg: 0.835, SGD: 0.8024, SVC: 0.7808")

MNB_f1 = f1_scores[0]
BNB_f1 = f1_scores[1]
LogReg_f1 = f1_scores[2]
SGD_f1 = f1_scores[3]
SVC_f1 = f1_scores[4]

Accuracy = [MNB_f1,BNB_f1,LogReg_f1,SGD_f1,SVC_f1]
Labels = ['MNB', 'BNB', 'LogReg', 'SGD' , 'SVC']
Accuracy_pos = np.arange(len(Labels))

plt.bar(Accuracy_pos, Accuracy)
plt.xticks(Accuracy_pos, Labels)
plt.title('Comparing the accuracy of each model')
plt.show()



'''

    f1 scores for the different models: 
    MNB: 0.845, BNB: 0.8447999, LogReg: 0.835, SGD: 0.8024, SVC: 0.7808

'''


'''
    Finally to get a good gauge of the implementation of NLTK, we model a a hybrid between all of 
    the classifiers using an Ensemble Model. 

    Ensemble Model: 

    An ensemble model combines the predictions (take votes) from each of the above models for each 
    review and uses the majority vote as the final prediction.


    For more information: 

    https://medium.com/data-science-in-your-pocket/ensemble-models-in-machine-learning-d429c988e866

    Due to the time it took to train each model (8-13 minutes on my computer) we will pickle the
    trainging process. 

'''


import pickle
from ensemble import EnsembleClassifier

# Load all classifiers from the pickled files 

# function to load models given filepath
def load_model(file_path): 
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier



# Original Naive Bayes Classifier
ONB_Clf = load_model('pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier 
MNB_Clf = load_model('pickled_algos/MNB_clf.pickle')


# Bernoulli  Naive Bayes Classifier 
BNB_Clf = load_model('pickled_algos/BNB_clf.pickle')

# Logistic Regression Classifier 
LogReg_Clf = load_model('pickled_algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model('pickled_algos/SGD_clf.pickle')


# Initializing the ensemble classifier 
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# List of only feature dictionary from the featureset list of tuples 
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]

# Function to do classification a given review and return the label a
# and the amount of confidence in the classifications
def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)



#Sentiment Analyzer for Movie Reviews -> Test Case :: 
# Avengers End Game (Action) 

text_a = '''What's missing from "Endgame" is the free play of imagination, the liberation of speculation, the meandering paths and loose ends that start in logic and lead to wonder.'''
text_b = '''The only complaint about Avengers: Endgame is that it raises the bar so high that there may well never be a superhero movie to match it.'''
text_c = '''The film is almost three hours long; long enough to tie up loose ends, as well as give foundation to a new generation of Avengers. When all is said and done, this was the hardest goodbye in the Marvel Universe.'''
sentiment(text_a), sentiment(text_b), sentiment(text_c)

#Reviews for Get Smart (Comedy)

text_d = '''As a reworking of one of the great 1960s TV comedies, you'd think being funny would be its main goal. But you would be wrong. Very, very wrong'''
text_e = '''The film is about as funny as a 3am cold call, and if you are a critic you cannot slam down the phone before you have heard it out.'''
text_f = '''Get Smart doesn't have a lot of laugh out loud moments, but it does have plenty of chuckles, which is just enough to make it miles better than the other new comedy.'''
sentiment(text_d, text_e, text_f)




# %%
