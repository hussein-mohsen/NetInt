from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn import metrics
from helper_functions import read_dataset

from collections import Counter

import numpy as np
from builtins import str

def main():
    dataset_name = 'diabetes'
    ml_algo = 'rf'
    one_hot_encoding = False
    
    D = read_dataset(dataset_name=dataset_name, one_hot_encoding=one_hot_encoding)
    X_tr, Y_tr = D.train.points, D.train.labels
    X_ts, Y_ts = D.test.points, D.test.labels

    n_classes = len(np.unique(Y_tr))
    
    print('Training Label Distribution: ', str(report_label_distribution(Y_tr)))
    print('Test Label Distribution: ', str(report_label_distribution(Y_ts)))
    
    if ml_algo == 'rf':
        clf = RandomForestClassifier(n_estimators=100)
    elif ml_algo == 'svm':
        clf = svm.SVC(gamma='scale')
    
    clf.fit(X_tr, Y_tr)
            
    Y_tr_pred = clf.predict(X_tr)
    Y_ts_pred = clf.predict(X_ts)
    
    print('Training Accuracy: ', metrics.accuracy_score(Y_tr, Y_tr_pred))
    print('Test Accuracy: ', metrics.accuracy_score(Y_ts, Y_ts_pred))

    if(n_classes == 2):
        print('Training Precision: ', metrics.precision_score(Y_tr, Y_tr_pred))
        print('Test Precision: ', metrics.precision_score(Y_ts, Y_ts_pred))
        
        print('Training AUC: ', metrics.roc_auc_score(Y_tr, Y_tr_pred))
        print('Test AUC: ', metrics.roc_auc_score(Y_ts, Y_ts_pred))

def report_label_distribution(Y):
    dist_str = ''
    if Y is not None:
        dist_str = str(Y.shape) + '-'
        
        label_dist = Counter(Y)
        total = sum(label_dist.values())

        for k in label_dist.keys():
            normalized_value = label_dist[k] / total
            label_dist[k] = round(normalized_value * 100, 2)
            dist_str = dist_str + str(label_dist[k]) + "-"
            
        dist_str = dist_str[:-1] + '%'

    return dist_str

main()