
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


###dictionary to dataframe    
fp= pd.DataFrame(data_dict)
fp =fp.transpose()

fp.poi = fp.poi.astype(int)
fp.bonus =  fp.bonus.astype(float)
fp.deferral_payments = fp.deferral_payments.astype(float)   
fp.deferred_income = fp.deferred_income.astype(float)          
fp.director_fees = fp.director_fees.astype(float)       
fp.email_address = fp.email_address.astype(str)       
fp.exercised_stock_options =  fp.exercised_stock_options.astype(float)     
fp.expenses = fp.expenses.astype(float)            
fp.from_messages = fp.from_messages.astype(float)           
fp.from_poi_to_this_person = fp.from_poi_to_this_person.astype(float)     
fp.from_this_person_to_poi = fp.from_this_person_to_poi.astype(float)    
fp.loan_advances = fp.loan_advances.astype(float)  
fp.long_term_incentive = fp.long_term_incentive.astype(float)  
fp.other = fp.other.astype(float)                  
fp.restricted_stock = fp.restricted_stock.astype(float)     
fp.restricted_stock_deferred =fp.restricted_stock_deferred.astype(float)  
fp.salary = fp.salary.astype(float)   
fp.shared_receipt_with_poi = fp.shared_receipt_with_poi.astype(float) 
fp.to_messages = fp.to_messages.astype(float)   
fp.total_payments = fp.total_payments.astype(float)     
fp.total_stock_value =fp.total_stock_value.astype(float) 
fp.email_address = fp.email_address.astype('str')

fp = fp.drop(['email_address'], axis=1)

### Task 2: Remove outliers
### How I arrive with outliers documented in Hernando_Notes_on_Enron_Project.ipynb
outliers = fp[(np.abs(fp.salary-fp.salary.mean())>(3*fp.salary.std()))] 
outliers
#remove outliers
fp = fp[~(np.abs(fp.salary-fp.salary.mean())>(3*fp.salary.std()))] 


### Task 3: Create new feature(s)
fp['prop_to_poi']=  fp['from_this_person_to_poi'] / fp['to_messages']
fp['prop_from_poi']=  fp ['from_poi_to_this_person']/fp['from_messages']

#DATA Cleansing
#drop variables with NAN more than 50%
fp = fp.drop(['long_term_incentive','deferred_income','deferral_payments','restricted_stock_deferred','director_fees','loan_advances','from_this_person_to_poi','to_messages', 'from_poi_to_this_person','from_messages' ], axis=1)

#impute 0 for financial features
fp.salary.fillna(value=0, inplace=True) 
fp.total_payments.fillna(value=0, inplace=True) 
fp.bonus.fillna(value=0, inplace=True) 
fp.total_stock_value.fillna(value=0, inplace=True) 
fp.expenses.fillna(value=0, inplace=True) 
fp.exercised_stock_options.fillna(value=0, inplace=True) 
fp.other.fillna(value=0, inplace=True) 
fp.restricted_stock.fillna(value=0, inplace=True) 

#impute mean for email features
fp["shared_receipt_with_poi"].fillna(fp["shared_receipt_with_poi"].mean(), inplace=True)
fp["prop_to_poi"].fillna(fp["prop_to_poi"].mean(), inplace=True)
fp["prop_from_poi"].fillna(fp["prop_from_poi"].mean(), inplace=True)

#feature rescaling
fp = (fp - fp.min()) / (fp.max() - fp.min())

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### How I arrive with these features documented in Hernando_Notes_on_Enron_Project.ipynb


#correlation analysis
fp.corr()['poi']

### here we found that the feature engineering for prop_to_poi and prop_from poi also getting higher correlation, compared by individual features to_messages, from_this_person_to_poi, from_messages , and from_poi_to_this_person 
#this proven effective so we will also drop

#return dataframe into dictionary
fp = fp.to_dict(orient="index")
# First using all available features and us various classifier to test the best algorythm
features_list = ['poi','bonus','exercised_stock_options','expenses','other','restricted_stock','salary','shared_receipt_with_poi','total_payments','total_stock_value','prop_to_poi','prop_from_poi' ] # You will need to use more features


from tester import test_classifier

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)
clf2= GaussianNB()
clf3= RandomForestClassifier(random_state=42)
clf4= SGDClassifier(random_state=42)

test_classifier(clf, fp, features_list, folds = 1000)
test_classifier(clf2, fp, features_list, folds = 1000)
test_classifier(clf3, fp, features_list, folds = 1000)
test_classifier(clf4, fp, features_list, folds = 1000)

### Result from initial classifier using all features
### DecisionTreeClassifier Accuracy: 0.80840	Precision: 0.26327	Recall: 0.24300	F1: 0.25273	F2: 0.24680 #BEST higher overall precision, recall& F1
### GaussianNB             Accuracy: 0.83920	Precision: 0.32890	Recall: 0.19800	F1: 0.24719	F2: 0.21512
### RandomForestClassifier Accuracy: 0.86073	Precision: 0.42811	Recall: 0.13250	F1: 0.20237	F2: 0.15373
### SGDClassifier          Accuracy: 0.52980	Precision: 0.10665	Recall: 0.34250	F1: 0.16265	F2: 0.23747

# Using Decision Tree Classifier to find attributes of importance
fp = pd.DataFrame(fp)
fp = fp.transpose()

X = fp.drop(['poi'], axis=1)
y = fp['poi']

## crete the features_train, features_test,labels_train,labels_tests
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=42)

## fit the decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(features_train,labels_train)

## Features of importance
col_values = X.columns.values
importance= clf.feature_importances_
ip= pd.DataFrame(importance,col_values)
ip.columns = ['Importance']
ip= ip.sort_values(by=('Importance'), ascending=False)
ip

ip.Importance.plot(kind='bar',figsize=(16, 8), alpha=0.95)
plt.title("Degree of Importance by variable")

### Task 1: Select what features you'll use.first attribute must be poi
### based on degree of importance we will do cutoff from degree of importance less than 0.05 , so there are 6 variables to be used as prediction model:
###1. Expenses
###2.Bonus
###3.other
###4.prop_to_poi
###5.restricted_stock
###6.total_stock_value

### In summary we made the decision of variables,based on
### first,we eliminate features that have NAN values more than 50%, then we create nconduct correlation analysis 
### second, we create new variables, and conduct correlation analysis,then we eliminate the underlying variables which are individual features to_messages, from_this_person_to_poi, from_messages , and from_poi_to_this_person. This is because the new variables proven that they have stronger correlation with poi than individual features.
###Third, we conduct different classifiers , and decided to use decision tree classifier of for features of importance analysis based on the precision, recall and f1
### fourth, we conduct features of importance analysis and do cut off for degree of importance less than 0.05
### that's how we arrive with this 6 features.

features_list = ['poi','expenses','bonus','other','prop_to_poi','restricted_stock','total_stock_value']


#return dataframe into dictionary
fp = fp.to_dict(orient="index")


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf = DecisionTreeClassifier(random_state=42 )
clf2= GaussianNB()
clf3= RandomForestClassifier(random_state=42)
clf4= SGDClassifier(random_state=42)

test_classifier(clf, fp, features_list, folds = 1000)
test_classifier(clf2, fp, features_list, folds = 1000)
test_classifier(clf3, fp, features_list, folds = 1000)
test_classifier(clf4, fp, features_list, folds = 1000)

## Result from initial classifier using 7 selected features
#I ended up choosing Naives Bayes and Decision tree, as both give the default much better, yet similar performance compared to other classifier that I tried. The performance default for each of the algorithm are as follows:
 
#1. DecisionTreeClassifier() ##Accuracy: 0.81407	Precision: 0.29421	Recall: 0.28200	F1: 0.28798	2nd BEST!
#2. GaussianNB()             ##Accuracy: 0.85200	Precision: 0.41947	Recall: 0.28650	F1: 0.34046 BEST!
#3. RandomForestClassifier() ##Accuracy: 0.85967	Precision: 0.41758	Recall: 0.13300	F1: 0.20174	
#4. SGDClassif               ##Accuracy: 0.81613	Precision: 0.28898	Recall: 0.25950	F1: 0.27345	

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Extract features and labels from dataset for local testing
my_dataset = fp

data = featureFormat(my_dataset, features_list, sort_keys = True)
y, X = targetFeatureSplit(data)
X = np.array(X)
y = np.array(y)


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(grid, X, y, cv):
    nested_score = cross_val_score(grid, X=X, y=y, cv=cv, n_jobs=-1)
    print "Nested f1 score: {}".format(nested_score.mean())

    grid.fit(X, y)    
    print "Best parameters: {}".format(grid.best_params_)

    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid.best_estimator_.fit(X_train, y_train)
        pred = grid.best_estimator_.predict(X_test)

        cv_accuracy.append(accuracy_score(y_test, pred))
        cv_precision.append(precision_score(y_test, pred))
        cv_recall.append(recall_score(y_test, pred))
        cv_f1.append(f1_score(y_test, pred))

    print "Mean Accuracy: {}".format(np.mean(cv_accuracy))
    print "Mean Precision: {}".format(np.mean(cv_precision))
    print "Mean Recall: {}".format(np.mean(cv_recall))
    print "Mean f1: {}".format(np.mean(cv_f1))


### Cross-validation
sss = StratifiedShuffleSplit(n_splits=12, test_size=0.25, random_state=42)

SCALER = [None, StandardScaler()]
SELECTOR__K = ['all']
REDUCER__N_COMPONENTS = [2, 4, 6]

### Gaussian Naives Bayes

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', GaussianNB())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS
}

gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)



### Decision Tree
CRITERION = ['gini', 'entropy']
SPLITTER = ['best', 'random']
MIN_SAMPLES_SPLIT = [ 2, 4, 6]
CLASS_WEIGHT = ['balanced', None]

### comment to perform a full hyperparameter search
# SCALER = [StandardScaler()]
# SELECTOR__K = [18]
# REDUCER__N_COMPONENTS = [2]
# CRITERION = ['gini']
# SPLITTER = ['random']
# MIN_SAMPLES_SPLIT = [8]
# CLASS_WEIGHT = ['balanced']
###################################################

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', DecisionTreeClassifier())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS,
    'classifier__criterion': CRITERION,
    'classifier__splitter': SPLITTER,
    'classifier__min_samples_split': MIN_SAMPLES_SPLIT,
    'classifier__class_weight': CLASS_WEIGHT
}

tree_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

evaluate_model(gnb_grid, X, y, sss)

evaluate_model(tree_grid, X, y, sss)

#The result from best estimator between Gaussian Naive Bayes and Decision Tree Classifier were: 

## GNB
#- Nested f1 score: 0.28126984127
#- Best parameters: {'reducer__n_components': 4, 'selector__k': 'all', 'scaler': None}
#- Mean Accuracy: 0.851724137931
#- Mean Precision: 0.388333333333
#- Mean Recall: 0.275
#- Mean f1: 0s.301507936508

## Decision Tree Classifier
#- Nested f1 score: 0.332106782107

#- Best parameters: {'reducer__n_components': 2, 'selector__k': 'all', 'scaler': None, 'classifier__min_samples_split': 8, 'classifier__class_weight': 'balanced', 'classifier__splitter': 'best', 'classifier__criterion': 'gini'}
#- Mean Accuracy: 0.834482758621
#- Mean Precision: 0.460606060606
#- Mean Recall: 0.575
#- Mean f1: 0.491240981241

#The best decision tree classifier is the best model with higher precision and recall and nester f1 score of 0.332


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
test_classifier(tree_grid.best_estimator_, my_dataset, features_list)

clf =  tree_grid.best_estimator_ 
#Test classifier showing precision and recall more than 0.3:
#- Accuracy: 0.79027	
#- Precision: 0.31693	
#- Recall: 0.49600	
#- F1: 0.38674
 
dump_classifier_and_data(clf, my_dataset, features_list)

