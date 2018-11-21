from preprocessing import KeywordCorpus, KeywordCorpusFactory

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
		kcf = KeywordCorpusFactory()
		assert True

	def test_keyword_corpus(self):

		kcf = KeywordCorpusFactory()
		assert kcf.keyword_corpus['fuck'] == 'Corpus of Keyword fuck does not exist.'

	def test_create_keyword_corpus(self):

		kcf = KeywordCorpusFactory()
		test_sentences = ['I am cool.', 'I am handsome.']
		kcf.create_keyword_corpus('I', test_sentences)
		assert kcf.keyword_corpus['I'] == [['am', 'cool.'], ['am', 'handsome.']]

