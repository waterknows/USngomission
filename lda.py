from gensim import corpora, models, similarities
print("Loading data")
# Initialize Parameters
corpus_filename = 'mission2.mm'
dict_filename   = 'mission2.dict'

for n in [25,50,100,200,300,500,1000]:
	lda_filename    = 'mission'+str(n)+'.lda'
	lda_params      = {'passes': 5, 'alpha': 'auto'}
	# Load the corpus and Dictionary
	corpus = corpora.MmCorpus(corpus_filename)
	dictionary = corpora.Dictionary.load(dict_filename)

	print("Running LDA with: %s  topics" % n)
	lda = models.LdaModel(corpus, id2word=dictionary,num_topics=n,passes=lda_params['passes'],alpha = lda_params['alpha'])
	lda.save(lda_filename)
	print("lda saved in %s " % lda_filename)