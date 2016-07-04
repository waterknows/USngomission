from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import string
import sys

#some codes are borrowed from https://github.com/rayidghani/magicloops/blob/master/magicloops.py

def select_classifiers():
    return ['LR','NB','DT','KNN','SVM','SGD','ET','RF','AB','GB']

clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

grid = {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        'RF':{'n_estimators': [1000], 'max_depth': [10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': { 'n_estimators': [10,100,1000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000]},
        'GB': {'n_estimators': [10,100,1000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
               }


def loop_clf(feature, label, models,k):
    feature=feature.fillna(value=0)
    msk = np.random.rand(len(feature)) < 0.9
    train = feature[msk]
    validate = feature[~msk]
    y_train=train[label]
    y_validate = validate[label]
    x_columns=[i for i in feature.columns if not i==label]
    x_train=train[x_columns]
    x_validate=validate[x_columns]
    print('train-test split finished','X_columns',x_train.columns,'X_length',len(x_train),'Y_distribution',y_train.describe())

    classifiers=[]
    accuracies=[]
    precisions=[]
    recalls=[]
    F1s=[]
    AUCs=[]
    parameters=[]
    ts=[]
    thrs=[]
    for index,clf in enumerate([clfs[x] for x in models]):
        parameter_values = grid[models[index]]
        for p in ParameterGrid(parameter_values):
            try:
                start = time.time()
                clf.set_params(**p)
                print("Running",clf)
                y_pred_probs = clf.fit(x_train, y_train).predict_proba(x_validate)[:,1]
                end = time.time()
                t=end - start
                threshold = np.sort(y_pred_probs)[::-1][int(k*len(y_pred_probs))]
                y_pred = np.asarray([1 if i >= threshold else 0 for i in y_pred_probs])
                #metrics
                accuracy = metrics.accuracy_score(y_validate, y_pred)
                precision = metrics.precision_score(y_validate, y_pred)
                recall = metrics.recall_score(y_validate, y_pred)
                F1 = metrics.f1_score(y_validate, y_pred)
                AUC = metrics.roc_auc_score(y_validate, y_pred)
                thr=str(k)
                print("Accuracy",accuracy)
                print("Precision",precision)
                print("Recall",recall)
                print("ROC_AUC",AUC)
                #append
                accuracies.append(accuracy)
                classifiers.append(clf)
                parameters.append(p)
                precisions.append(precision)
                recalls.append(recall)
                F1s.append(F1)
                AUCs.append(AUC)
                ts.append(t)
                thrs.append(thr)
            except IndexError as e:
                accuracies.append(e)
                classifiers.append(e)
                parameters.append(e)
                precisions.append(e)
                recalls.append(e)
                AUCs.append(e)
                ts.append(e)
                thrs.append(e)
                continue

    df=pd.DataFrame(data={'classifier':classifiers,'threshold':thrs,'parameter':parameters,'time':ts,'accuracy':accuracies,'precision':precisions,'recall':recalls,'F1':F1s,'AUC':AUCs})
    return df


def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

def run(inp, out):
    '''
    Loop through all models.
    '''
    fea = pd.read_table(sys.argv[1],index_col='EIN')
    result=loop_clf(feature=fea, label='CONTRIBUTION', models=['KNN','NB','DT','RF','LR'],k=0.5)
    result.to_csv(out,sep='\t')
    print ("Wrote file to {}".format(out))



if __name__ == '__main__':
    print('Usage: python3 model.py <Input> <Output>')
    run(sys.argv[1],sys.argv[2])
