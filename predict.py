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