from gensim import corpora, models
import pyLDAvis.gensim
print('Loading model')
corpus = corpora.MmCorpus('mission2.mm')
dictionary = corpora.Dictionary.load ('mission2.dict')
lda = models.LdaModel.load('mission50.lda')
print('Building visualization')
mission_data =  pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(mission_data)
