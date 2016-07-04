from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import refrom gensim import corpora, models
import pyLDAvis.gensim
import nltk
import string
import pandas as pd
from collections import defaultdict
from gensim import corpora, models, similarities
from time import time
from nltk.tokenize import RegexpTokenizer

print("Loading dataset...")
t0 = time()
data = pd.read_table('ngo2.txt',index_col='EIN')
tokenizer = RegexpTokenizer(r'\w+')
data['COUNT']=data['MISSION'].apply(lambda x: len(tokenizer.tokenize(str(x))))
documents = data[data['COUNT']>20]['MISSION'].tolist()
print("Number of Organizations",len(documents),"done in %0.3fs." % (time() - t0))
stoplist= ["mission","object","goal","purpose","not-for-profit","nonprofit","ngo","organization","community",
"program","activity","service","education","training","volunteer","support","people","need"]
print("Generating distribution")
wnl = WordNetLemmatizer()
fdist=nltk.FreqDist(wnl.lemmatize(word) for doc in documents for word in tokenizer.tokenize(doc))
tf1=fdist.hapaxes()
def nouns(sent):
	nouns=[token for token,pos in pos_tag(word_tokenize(sent)) if pos.startswith('N')]
	return nouns
print("Tokenizing, extrating nouns, removing non-informative words and lemmatizing")
t0 = time()
def tokenize(doc):
	tokens = [word for sent in nltk.sent_tokenize(doc) for word in nouns(sent)] # first tokenize by sentence, then by word to extract nouns
	wnl = WordNetLemmatizer()
	singulars=[wnl.lemmatize(t) for t in tokens]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in singulars:
		if (re.search('[a-z]', token) and token not in stoplist and token not in tf1):
			filtered_tokens.append(token)
	return filtered_tokens
documents = [ tokenize(doc.lower()) for doc in documents ]
print("Pre-processing done in %0.3fs." % (time() - t0))
print("Generating dictionary and corpus")
# Sort words in documents
for doc in documents:
	doc.sort()
dictionary = corpora.Dictionary(documents) # Build a dictionary where for each document each word has its own id
dictionary.compactify() 
dictionary.save('mission2.dict') # and save the dictionary for future use
corpus = [dictionary.doc2bow(doc) for doc in documents] # Build the corpus: vectors with occurence of each word for each document
corpora.MmCorpus.serialize('mission2.mm', corpus) # and save in Market Matrix format