import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
import numpy as np
from sklearn.linear_model import LogisticRegression

def rfparameters(df,label,clf):
    features=np.array(df.ix[:, df.columns != label].describe().keys())
    print('Running RF')
    clf.fit(df[features], df[label])
    print('Plotting and Recording')
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[:10]
    padding = np.arange(10) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    best_features = features[sorted_idx][::-1]
    ddf=pd.DataFrame(data={'Top Features by RF': best_features})
    return pl.savefig('importanceRF.png'), ddf.to_csv('importanceRF.txt',sep='\t')
def lrparameters(df, label, clf):
    features=[i for i in df.columns if not i==label]
    print('Running LR')
    clf.fit(df[features], df[label])
    print('Recording')
    ddf=pd.DataFrame(clf.coef_,columns=df[features].columns).transpose()
    return ddf.to_csv('importanceLR.txt',sep='\t')
clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=100, max_features='sqrt', 
            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=5,min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,warm_start=False)
clf2=LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
print('Reading data')
df = pd.read_table('feature1000.txt',index_col='EIN')
df = df.fillna(value=0)
label = 'CONTRIBUTION'
'''
rfparameters(df,label,clf)
'''
lrparameters(df,label,clf2)