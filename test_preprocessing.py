import pytest

from preprocessing import KeywordCorpus
from preprocessing import KeywordCorpusFactory
from preprocessing import KeywordCorpusIterator
from test_embedding import WiKiCorpusIterator

class TestKeywordCorpus():

	def test_init(self):
		kc = KeywordCorpus()
		assert True

	def test_set_and_get_item(self):
		kc = KeywordCorpus()
		kc['test'] = 'is good'
		assert kc['test'] == 'is good'

class TestKeywordCorpusIterator():

	def test_init(self):

		kc = KeywordCorpus()
		kc['test'] = set(['test is good', 'test is nice'])
		kc_iter = KeywordCorpusIterator(kc)

		count_token = 0
		for tokens in kc_iter:
			if not isinstance(tokens, list):
				assert False
			count_token += 1

		if count_token != 2:
			assert False

		count_token = 0
		kc_iter = KeywordCorpusIterator(kc, False)

		for tokens in kc_iter:
			if not isinstance(tokens, str):
				assert False
			count_token += 1

		if count_token != 2:
			assert False

class TestKeywordCorpusFactory():

	def test_init(self):

		keywords = ['test', 'test again']
		kcf = KeywordCorpusFactory(keywords)
		assert True

		del kcf;

		a_keyword = 'a keyword'
		with pytest.raises(Exception):
			kcf = KeywordCorpusFactory(a_keyword)
			
		keywords = [1,2,3,4,5]
		with pytest.raises(Exception):
			kcf = KeywordCorpusFactory(keywords)


	def test_keyword_corpus(self):

		kcf = KeywordCorpusFactory(['hello', 'world'])

		# 20181124 Hannah Chen, create keyword corpus
		keyword_corpus = kcf.create(
			['Hello World is our first program', 
			'Hello World is our last program']
		)
		assert keyword_corpus['hi'] == 'Corpus of Keyword hi does not exist.'

		# assert kcf.keyword_corpus['hi'] == 'Corpus of Keyword hi does not exist.'

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

	def test_add_keyword_corpus(self):

		start_keywords = ['not', 'so', 'important']
		keyword = 'Hello'
		corpus  = ['This is a Hello World Sample.', 'This is nothing']
		kcf = KeywordCorpusFactory(start_keywords)
		kcf.add_keyword_corpus(keyword, corpus)
		assert kcf.kc['Hello'] == set(corpus)

		new_corpus = ['This a new hello word sample']
		kcf.add_keyword_corpus(keyword, new_corpus)
		assert kcf.kc['Hello'] == set(corpus + new_corpus)

		# Test for list input which should be illegal
		with pytest.raises(Exception):
			kcf.add_keyword_corpus(['Hi again'], corpus)


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

