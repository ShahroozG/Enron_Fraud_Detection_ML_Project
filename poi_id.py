#!/usr/bin/python

# import sys
# sys.path.append("../tools/")

import pickle
import matplotlib.pyplot
import pandas
import copy
import numpy as np

# from feature_format import featureFormat, targetFeatureSplit
# from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### You will need to use more features; 

### SH: First I put all features except email_address because it's an string and it's not helpful
#features_list =  ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus',\
# 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses',\
# 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income',\
# 'long_term_incentive', 'from_poi_to_this_person', 'bonus_square', 'stock_square']

# 'shared_receipt_with_poi',
### SH: financial features only
features_list = ['poi','shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'total_payments',\
 'restricted_stock', 'other', 'restricted_stock_deferred', 'total_stock_value', 'expenses','salary', 'deferral_payments', \
 'loan_advances', 'director_fees', 'deferred_income','long_term_incentive', 'bonus_square', 'exercised_stock_square']

### Load the dictionary containing the dataset
path_1 = "C:/Users/Shahrooz/Desktop/P5-ML/ud120-projects/final_project/"
with open(path_1 + "final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### SH: removing the 'TOTAL' outlier
data_dict.pop("TOTAL", 0)
## These two other outliers removal doesn't cause better performance so I didn't remove them
#data_dict.pop("LAY KENNETH L", 0)
#data_dict.pop("SKILLING JEFFREY K", 0)

### Task 3: Create new feature(s)
## SH: New_Feature_1: Bonus^2; New_Feature_2: Exercised_stock_options^2
data_dict_new = copy.deepcopy(data_dict)

for person in data_dict_new:
    if data_dict_new[person]['bonus'] != "NaN":
        data_dict_new[person]['bonus_square'] = data_dict_new[person]['bonus']**2
    else:
        data_dict_new[person]['bonus_square'] = "NaN"
        
    if data_dict_new[person]['exercised_stock_options'] != "NaN":
        data_dict_new[person]['exercised_stock_square'] = data_dict_new[person]['exercised_stock_options']**2
    else:
        data_dict_new[person]['exercised_stock_square'] = "NaN"
        
    if data_dict_new[person]['salary'] != "NaN":
        data_dict_new[person]['salary_square'] = data_dict_new[person]['salary']**2
    else:
        data_dict_new[person]['salary_square'] = "NaN"

### Store to my_dataset for easy export below.
my_dataset = data_dict_new

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import StratifiedShuffleSplit


scaler = MinMaxScaler()
pca = PCA()
selector = SelectKBest()
classifier_NB = GaussianNB()
classifier_SVM = SVC(kernel='rbf', class_weight='balanced')
classifier_DT = tree.DecisionTreeClassifier()

### SH: I am creating my pipeline: Scaler ==> PCA ==> Selector ==> Classifier

### Naive Bayse
estimators_NB = [('scaling', scaler), ('reduce_dim', pca), ('selector', selector), ('classifier', classifier_NB)]
#estimators_NB = [('scaling', scaler), ('selector', selector), ('classifier', classifier_NB)]
params_NB = dict(reduce_dim__n_components=[8, 10, 12, 14], selector__k = [5, 8])
clf_NB = Pipeline(estimators_NB)
grid_search_NB = GridSearchCV(clf_NB, param_grid=params_NB, scoring='f1')

### SVM
estimators_SVM = [('selector', selector), ('classifier', classifier_SVM)]
params_SVM = dict(selector__k = [5, 8], classifier__C = [1, 10, 100, 1000, 10000])
clf_SVM = Pipeline(estimators_SVM)
grid_search_SVM = GridSearchCV(clf_SVM, param_grid=params_SVM, scoring='f1')

### Decision Tree
estimators_DT = [('reduce_dim', pca), ('selector', selector), ('classifier', classifier_DT)]
params_DT = dict(selector__k = [3, 5, 8], reduce_dim__n_components=[10, 12, 13], \
                 classifier__min_samples_split = [5, 10, 15, 25], classifier__criterion = ['gini', 'entropy'])
clf_DT = Pipeline(estimators_DT)
grid_search_DT = GridSearchCV(clf_DT, param_grid=params_DT, scoring='f1')

### SH: Cross Validation: Spliting data in training and testing sets
# Example starting point. Try investigating other evaluation techniques!
# this is so simple and prone to problems:
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Use another method for shuffeling data and splitting it to training and testing
cv = StratifiedShuffleSplit(labels, random_state = 42)
for train_index, test_index in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for i in train_index:
        features_train.append( features[i] )
        labels_train.append( labels[i] )
    for j in test_index:
        features_test.append( features[j] )
        labels_test.append( labels[j] )

### SH: Training all Classifiers and choose the best for each one; geeting Performance results for each of them
### SH: training NB
grid_search_NB.fit(features_train, labels_train)
clf_NB = grid_search_NB.best_estimator_
print clf_NB
pred_NB = clf_NB.predict(features_test)
precision_NB = precision_score(pred_NB, labels_test)
recall_NB = recall_score(pred_NB, labels_test)
f1_NB = 2 * (precision_NB * recall_NB)/(precision_NB + recall_NB)
print "Precision_NB: ", precision_NB
print "Recall_NB: ", recall_NB
print "F1_Score_NB: ", f1_NB

### SH: training SVM
### It creates error every time, so I comment out this part
#grid_search_SVM.fit(features_train, labels_train)
#clf_SVM = grid_search_SVM.best_estimator_
#print clf_SVM
#pred_SVM = clf_SVM.predict(features_test)
#precision_SVM = precision_score(pred, labels_test)
#recall_SVM = recall_score(pred_SVM, labels_test)
#f1_SVM = 2 * (precision_SVM * recall_SVM)/(precision_SVM + recall_SVM)
#print "Precision_SVM: ", precision_SVM
#print "Recall_SVM: ", recall_SVM
#print "F1_Score_SVM: ", f1_SVM


### SH: training DT
grid_search_DT.fit(features_train, labels_train)
clf_DT = grid_search_DT.best_estimator_
print clf_DT
pred_DT = clf_DT.predict(features_test)
precision_DT = accuracy_score(pred, labels_test)
recall_DT = recall_score(pred_DT, labels_test)
f1_DT = 2 * (precision_DT * recall_DT)/(precision_DT + recall_DT)
print "Precision_DT: ", precision_DT
print "Recall_DT: ", recall_DT
print "F1_Score_DT: ", f1_DT

index = np.argmax([f1_NB, f1_DT])
classifier = {0: 'Naive Bayse', 1: 'Decision Tree'}
print
print('Best classifier is {}'.format(classifier[index]))

clf = clf_NB

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)