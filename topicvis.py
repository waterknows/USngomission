from gensim import corpora, models
orpus = corpora.MmCorpus('mission2.mm')
dictionary = corpora.Dictionary.load ('mission2.dict')
lda = models.LdaModel.load('mission1000.lda')
print('RF importance')
for x in [265,336,175,951,187,58,460,795,689,14]:
	print(lda.print_topic(x))
print ('LR positive')
for x in [959,958,604,654,359]:
	print(lda.print_topic(x))
print ('LR negative')
for x in [845,305,98,333,415]:
	print(lda.print_topic(x))