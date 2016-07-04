from __future__ import print_function
from time import time
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

n_topics = 100
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    wnl = WordNetLemmatizer()
    singulars=[wnl.lemmatize(t) for t in filtered_tokens]
    stemmer = SnowballStemmer('english')
    stems = [stemmer.stem(t) for t in singulars]
    return stems


print("Loading dataset...")
t0 = time()
data = pd.read_table('ngo2.txt',index_col='EIN')
tokenizer = RegexpTokenizer(r'\w+')
data['COUNT']=data['MISSION'].apply(lambda x: len(tokenizer.tokenize(str(x))))
dataset = data[data['COUNT']>10]['MISSION']
print("Number of Organizations",len(dataset),"done in %0.3fs." % (time() - t0))

'''
# Generate tf-idf features.
print("Extracting tf-idf features...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, tokenizer=tokenize_and_stem, stop_words='english')
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(dataset)
print("done in %0.3fs." % (time() - t0))
'''
# Generate tf features.
print("Extracting tf features...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, tokenizer=tokenize_and_stem,stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(dataset)
print("done in %0.3fs." % (time() - t0))

'''
# Fit the NMF model
print("Fitting the NMF model with tf-idf features")
t0 = time()
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)
'''

# Fit the LDA model
print("Fitting LDA models with tf features")
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
doctopic=lda.fit_transform(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)

print("Saving document topic distribution")
pd.Dataframe(doctopic).to_csv('lda.txt',sep='\t')