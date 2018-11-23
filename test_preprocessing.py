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
		assert kcf.keyword_corpus['hi'] == 'Corpus of Keyword hi does not exist.'

	def test_create(self):

		keywords = [
			'anarchism', 'philosophy', 'autism', 'communication', 'autistic', 
			'family', 'studies', 'important', 'earth', 'incident']
		wc  = WiKiCorpusIterator(1000, return_str=True)
		kcf = KeywordCorpusFactory(keywords)
		keyword_corpus = kcf.create(wc)

		for k in keywords:
			if keyword_corpus[k] is None:
				assert False
		assert True

	def test_update(self):

		first_keywords = ['Hello', 'World']
		first_sentences = [
			['Hello World is our first program'], 
			['Hello World is our last program']
		]
		kcf = KeywordCorpusFactory(first_keywords)
		keyword_corpus = kcf.create(first_sentences)

		second_sentences = [
			['Hello World is a fantastic sample'],
			['Hello World is a fantastic example']
		]
		kcf.update(sentences=second_sentences)

		print("keyword_corpus['Hello']:")
		print(keyword_corpus['Hello'])
		assert keyword_corpus['Hello'] == set([
			'Hello World is our first program',
			'Hello World is our last program',
			'Hello World is a fantastic sample',
			'Hello World is a fantastic example'
		])

		assert keyword_corpus['World'] == set([
			'Hello World is our first program',
			'Hello World is our last program',
			'Hello World is a fantastic sample',
			'Hello World is a fantastic example'
		])		

		# assert keyword_corpus['Hello'] == [
		# 		['Hello', 'World', 'is', 'our', 'first', 'program'],
		# 		['Hello', 'World', 'is', 'our', 'last', 'program'],
		# 		['Hello', 'World', 'is', 'a', 'fantastic', 'sample'],
		# 		['Hello', 'World', 'is', 'a', 'fantastic', 'example']
		# ]

		# assert keyword_corpus['World'] == [
		# 		['Hello', 'World', 'is', 'our', 'first', 'program'],
		# 		['Hello', 'World', 'is', 'our', 'last', 'program'],
		# 		['Hello', 'World', 'is', 'a', 'fantastic', 'sample'],
		# 		['Hello', 'World', 'is', 'a', 'fantastic', 'example']
		# ]

		kcf.update(keywords=['program'])

		print("keyword_corpus['program']:")
		print(keyword_corpus['program'])

		assert keyword_corpus['program'] == set([
			'Hello World is our first program',
			'Hello World is our last program',
		])
		# assert keyword_corpus['program'] == [
		# 	['Hello', 'World', 'is', 'our', 'first', 'program'],
		# 	['Hello', 'World', 'is', 'our', 'last', 'program'],
		# 	# ['Hello', 'World', 'is', 'a', 'fantastic', 'sample'],
		# 	# ['Hello', 'World', 'is', 'a', 'fantastic', 'example']
		# ]

	# def test_create_keyword_corpus(self):

	# 	kcf = KeywordCorpusFactory()
	# 	test_sentences = ['I am cool.', 'I am handsome.']
	# 	kcf.create_keyword_corpus('I', test_sentences)
	# 	assert kcf.keyword_corpus['I'] == [['am', 'cool.'], ['am', 'handsome.']]

