from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso

from sklearn import svm

from sklearn import metrics
from helper_functions import read_dataset

from collections import Counter

import numpy as np
from builtins import str

def main():
    dataset_name = 'AML_1'
    classification = True
    one_hot_encoding = False
    scaling_type = 'standard'
    ml_algo = 'svm'
    
    
    D = read_dataset(dataset_name=dataset_name, one_hot_encoding=one_hot_encoding, scaling_type=scaling_type)
    X_tr, Y_tr = D.train.points, D.train.labels.flatten()
    X_ts, Y_ts = D.test.points, D.test.labels.flatten()

    n_classes = len(np.unique(Y_tr))
        
    if ml_algo == 'rf':
        clf = RandomForestClassifier(n_estimators=100)
    elif ml_algo == 'svm':
        clf = svm.SVC(gamma='scale')
    elif ml_algo == 'lasso':
        clf = Lasso(alpha=0.1)
        
    clf.fit(X_tr, Y_tr)
            
    Y_tr_pred = clf.predict(X_tr)
    Y_ts_pred = clf.predict(X_ts)
    
    if classification:
        print('Training Label Distribution: ', str(report_label_distribution(Y_tr)))
        print('Test Label Distribution: ', str(report_label_distribution(Y_ts)))
    
        print('Training Accuracy: ', metrics.accuracy_score(Y_tr, Y_tr_pred))
        print('Test Accuracy: ', metrics.accuracy_score(Y_ts, Y_ts_pred))
    
        if(n_classes == 2):
            print('Training Precision: ', metrics.precision_score(Y_tr, Y_tr_pred))
            print('Test Precision: ', metrics.precision_score(Y_ts, Y_ts_pred))
            
            print('Training AUC: ', metrics.roc_auc_score(Y_tr, Y_tr_pred))
            print('Test AUC: ', metrics.roc_auc_score(Y_ts, Y_ts_pred))
    else:
        print('Training MSE: ', np.square(Y_tr - Y_tr_pred).mean())
        print('Test MSE: ', np.square(Y_ts - Y_ts_pred).mean())

        print('Training R^2: ', metrics.r2_score(Y_tr, Y_tr_pred))
        print('Test R^2: ', metrics.r2_score(Y_ts, Y_ts_pred))

def report_label_distribution(Y):
    dist_str = ''
    if Y is not None:
        dist_str = str(Y.shape) + '-'
        label_dist = Counter(Y.flatten())
        total = sum(label_dist.values())

        for k in label_dist.keys():
            normalized_value = label_dist[k] / total
            label_dist[k] = round(normalized_value * 100, 2)
            dist_str = dist_str + str(label_dist[k]) + "-"
            
        dist_str = dist_str[:-1] + '%'

    return dist_str

main()