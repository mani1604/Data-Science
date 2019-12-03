"""
NAME:           sms_dataset_svc.py
AUTHOR:         Manish Sharma
VERSION:        1.0
CREATED:        November 21, 2019

Python 2.7.16 and scikit-learn v0.21.3.

Python code to read the sms dataset containing spam and ham labelled messages.
Performed tf and tf-idf vectorizations using CountVectorizer() and TfidfVectorizer() in scikit-learn.
5-fold cross-validation is used with SVC() classifier in stratified manner.
Results are reported for tf and tf-idf vectors with combinations of parameters of SVC():
 - kernel: linear, rbf, sigmoid
 - C parameter values: 0.1, 1, 10
 - random_state: 13
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict

# reading the dataset using pandas
# read the 'tab' separated dataset and create a Data Frame
# providing the data frame cloumn names as 'label' and 'message'
data_frame = pd.read_csv('sms_dataset.csv', sep='\t', header=None, names=['label', 'message'])

# labeling spam = 1 and ham = 0
data_frame['label'] = data_frame.label.map({'spam': 1, 'ham': 0})

# tf and tf-idf vectorizations using CountVectorizer() and TfidfVectorizer()
# using default values for all vectorizer parameters
# using CountVectorizer
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(data_frame['message'])

# using TfidfVectorizer
tfidf_vect = TfidfVectorizer()
X_tfidf = tfidf_vect.fit_transform(data_frame['message'])

# stratified folds 5-fold
k_fold_statified = StratifiedKFold(n_splits=5, random_state=13)

# Calling the SVC() classifier
c_param_list = [0.1, 1, 10]
kernels_list = ['linear', 'rbf', 'sigmoid']
rand_state = 13

res_dict = OrderedDict()
pre = list()
rec = list()
f1 = list()

vectorizations = ['tf', 'tf-idf']

for vectorization in vectorizations:
    for kernel in kernels_list:
        for c_param in c_param_list:

            if vectorization == 'tf':
                X = X_counts
            elif vectorization == 'tf-idf':
                X = X_tfidf

            svc_clf = SVC(C=c_param, kernel=kernel, random_state=13)

            # Using 5-fold cross-validation
            rec_marco_scores = cross_val_score(svc_clf, X, data_frame['label'],
                                               cv=k_fold_statified, scoring='recall_macro')
            pre_marco_scores = cross_val_score(svc_clf, X, data_frame['label'],
                                               cv=k_fold_statified, scoring='precision_macro')
            f1_marco_scores = cross_val_score(svc_clf, X, data_frame['label'],
                                              cv=k_fold_statified, scoring='f1_macro')

            # Finding the mean of macroaveraged scores
            # and rounding them to 5 digits after decimal
            pre_mean_rounded = round(pre_marco_scores.mean(), 5)
            rec_mean_rounded = round(rec_marco_scores.mean(), 5)
            f1_mean_rounded = round(f1_marco_scores.mean(), 5)

            # preparing the data to print for each parameter combination
            row = '{v} , C={c} , {k}'.format(v=vectorization, c=c_param, k=kernel)
            res_dict[row] = [pre_mean_rounded, rec_mean_rounded, f1_mean_rounded]
            pre.append(pre_mean_rounded)
            rec.append(rec_mean_rounded)
            f1.append(f1_mean_rounded)

# using pandas to print the values in a table
cols = ['Macro Pre', 'Macro Rec', 'Macro F1']
res_table = pd.DataFrame.from_dict(res_dict, orient='index', columns=cols)
print 'Results for tf and tf-idf vectors using scikit-learn with the following combinations of parameters:'
print ' - linear, rbf and sigmoid kernel'
print ' - C parameter value from the list [0.1, 1, 10]'
print ' - random_state=13\n'
print res_table

# printing best P, R and F
print '\nThe best P, R, F1 values are:'
print 'Best Precision: {p}'.format(p=max(pre))
print 'Best Recall: {r}'.format(r=max(rec))
print 'Best F1: {f}'.format(f=max(f1))
