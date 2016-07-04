import numpy as np
import pandas as pd
import sys
import math
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from time import time

def generate_features(data,topics):
    features = pd.DataFrame(index = data.index)
    features = generate_label(data, features, 288903.5)
    features = generate_dummies(data, features)
    features = generate_vectors(features,features,topics)
    return features
def generate_vectors(data,features,n):
    corpus = corpora.MmCorpus('mission2.mm')
    dictionary = corpora.Dictionary.load ('mission2.dict')
    lda = models.LdaModel.load('mission'+str(n)+'.lda')
    print('mission'+str(n)+'.lda is loaded')
    doc = [dict(lda[x]) for x in corpus]
    calc=pd.DataFrame(doc,index = data.index)
    features=features.join(calc)
    return features
def generate_dummies(data,features):
    var=['NTEE']
    for v in var:
        calc = pd.DataFrame(index=data.index)
        new = data[data[v].notnull()][v]
        calc = calc.join(new, how = 'left')
        if v=='NTEE':
            calc[v] = v[:4]+'_'+data[data[v].notnull()][v].astype(str).str.strip().str.get(0)
        else:
            calc[v] = v[:4]+'_'+data[data[v].notnull()][v].astype(str).str.strip()      
        calc.fillna(value = v+'_MISSING', inplace = True)
        rv = pd.get_dummies(calc[v])
        features=features.join(rv)
    print('dummies generated')
    return features
def generate_label(data,features,cap):
    calc = pd.DataFrame(index=data.index)
    calc['CONTRIBUTION']=data['CONTRIBUTION'].apply(lambda x: 1 if x > cap else 0)
    print ('contributions label generated')
    return features.join(calc)
def run(inp, out,topics):
    print("Loading dataset...")
    t0 = time()
    data = pd.read_table(inp,index_col='EIN')
    tokenizer = RegexpTokenizer(r'\w+')
    data['COUNT']=data['MISSION'].apply(lambda x: len(tokenizer.tokenize(str(x))))
    data = data[data['COUNT']>20]
    print("Number of Organizations",len(data),"done in %0.3fs." % (time() - t0))
    features = generate_features(data,topics)
    features.to_csv(out,sep='\t')
    print ("Wrote file to {}".format(out))
if __name__ == "__main__":
    print ("This program produces all features for model generation.")
    print ("It will write the file features data to a separate file.")
    print ("Usage: python feature.py <Input> <Output> <number of topics>")  
    run(sys.argv[1], sys.argv[2], sys.argv[3])