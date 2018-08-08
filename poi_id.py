#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from tester import dump_classifier_and_data
from tester import main

# Function to compute ratio of poi messages to all messages 
def computeFraction( poi_messages, all_messages ):
    if poi_messages == "NaN" or all_messages == "NaN" or all_messages == 0:
        return 0.0
    else:
        return float(poi_messages)/float(all_messages)
    
features_list = ['poi', 'bonus', 'other', 'expenses', 'total_payments', 'restricted_stock', 'total_stock_value',
                 'long_term_incentive', 'exercised_stock_options', 'shared_receipt_with_poi',
                 'from_this_person_to_poi', 'from_poi_to_this_person']

# Loading the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
        
# Outlier cleanup
data_dict.pop("TOTAL", 0)
data_dict.pop("BAXTER, JOHN C", 0)
data_dict.pop("DERRICK JR., JAMES V", 0)
data_dict.pop("FREVERT, MARK A", 0)
data_dict.pop("PAI, LOU L", 0)
data_dict.pop("WHITE JR, THOMAS E", 0)

# Creation of new features "fraction_from_poi" and "fraction_to_poi"
for name in data_dict:
    data_point = data_dict[name]
    fraction_from_poi = computeFraction(data_point["from_poi_to_this_person"], data_point["to_messages"])
    fraction_to_poi = computeFraction(data_point["from_this_person_to_poi"], data_point["from_messages"])
    data_point["fraction_from_poi"] = float("{0:.2f}".format(fraction_from_poi))
    data_point["fraction_to_poi"] = float("{0:.2f}".format(fraction_to_poi))
features_list += ["fraction_from_poi", "fraction_to_poi"]

my_dataset = data_dict

# Extraction of features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

true_pos_count=0
false_pos_count=0
false_neg_count=0
true_neg_count=0

# Validation using StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 1000, random_state = 0)
for train_index, test_index in sss:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for training_values in train_index:
        features_train.append(features[training_values])
        labels_train.append(labels[training_values])
    for testing_values in test_index:
        features_test.append(features[testing_values])
        labels_test.append(labels[testing_values])
            
    # Creation of a pipeline comprising scaler, selector, and classifier
    scaler = MinMaxScaler()            
    selector = SelectKBest(score_func=f_classif, k=12)
    classifier = AdaBoostClassifier(n_estimators=8, random_state=0)
    clf = Pipeline(steps=[('scaler', scaler), ('selector', selector), ('classifier', classifier)])
    
    # Fitting the pipeline and predicting test labels
    clf.fit(features_train, labels_train)
    labels_test_pred = clf.predict(features_test)

    # Counting metrics for computing precision and recall     
    for i in range(len(labels_test)):
        if labels_test[i]==1 and labels_test_pred[i]==1:
            true_pos_count+=1
        else:
            if labels_test[i]==0 and labels_test_pred[i]==1:
                false_pos_count+=1
            else:
                if labels_test[i]==1 and labels_test_pred[i]==0:
                    false_neg_count+=1
                else:
                    if labels_test[i]==0 and labels_test_pred[i]==0:
                        true_neg_count+=1

# Printing features selected by SelectKBest
feature_idx = clf.named_steps['selector'].get_support()
print "Features used:"
for i in range(len(feature_idx)):
    if feature_idx[i]:
        print features_list[i], round(clf.named_steps['selector'].scores_[i],2)

# Computing and printing precision and recall         
precision=0.
recall=0.
if (true_pos_count+false_pos_count)!=0:
    precision = float(true_pos_count)/(true_pos_count+false_pos_count)
    
if (true_pos_count+false_neg_count)!=0:
    recall = float(true_pos_count)/(true_pos_count+false_neg_count)

print "\nPerformance metrics"
print "Precision: %.2f" %precision
print "Recall: %.2f" %recall

# Dumping your classifier, dataset, and features_list to enable testing
dump_classifier_and_data(clf, my_dataset, features_list)
