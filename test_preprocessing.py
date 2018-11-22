from preprocessing  import KeywordCorpus, KeywordCorpusFactory
from test_embedding import WiKiCorpusIterator

class TestKeywordCorpus():

	def test_init(self):
		kc = KeywordCorpus()
		assert True

	def test_set_and_get_item(self):
		kc = KeywordCorpus()
		kc['test'] = 'is good'
		assert kc['test'] == 'is good'

class TestKeywordCorpusFactory():

	def test_init(self):
		keywords = ['test', 'test again']
		kcf = KeywordCorpusFactory(keywords)
		assert True

	def test_keyword_corpus(self):

		kcf = KeywordCorpusFactory(['hello', 'world'])
		assert kcf.keyword_corpus['fuck'] == 'Corpus of Keyword fuck does not exist.'


	def test_create(self):

		keywords = [
			'anarchism', 'philosophy', 'autism', 'communication', 'autistic', 
			'family', 'studies', 'important', 'earth', 'incident']
		wc  = WiKiCorpusIterator(1000, return_str=True)
		kcf = KeywordCorpusFactory(keywords)
		keyword_corpus = kcf.create(wc)

		for k in keywords:
			if k not in keyword_corpus:
				assert False
			else:
				for tokens in keyword_corpus[k]:
					if k in tokens:
						assert False
		assert True

	# def test_create_keyword_corpus(self):

	# 	kcf = KeywordCorpusFactory()
	# 	test_sentences = ['I am cool.', 'I am handsome.']
	# 	kcf.create_keyword_corpus('I', test_sentences)
	# 	assert kcf.keyword_corpus['I'] == [['am', 'cool.'], ['am', 'handsome.']]

