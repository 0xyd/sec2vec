import os
import gc
import multiprocessing as mp

import pytest
import numpy as np
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

from embedding import SecWord2Vec
from embedding import SecFastText

sample_sentences = [
	['This', 'is', 'a', 'hello', 'world', 'example.'],
	['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
	['Is', 'this', 'a', 'hello', 'world', 'example'],
	['Are', 'there', 'any', 'examples']
]

wiki_keywords = [
	'anarchism', 'philosophy', 'autism', 'communication', 'autistic', 
	'family', 'studies', 'important', 'earth', 'incident',
	'is'
]

class WiKiCorpusIterator():

	def __init__(self, limit, return_str=True):

		wiki_dump_path = datapath(
			os.path.abspath('enwiki-latest-pages-articles1.xml-p10p30302.bz2'))
		wiki_corpus = WikiCorpus(wiki_dump_path)

		self.limit = limit
		self.count = 0
		self.wiki_iter = wiki_corpus.get_texts()
		self.return_str = return_str

	def __iter__(self): return self

	def __next__(self):

		global wiki_corpus

		if self.count == self.limit-1:
			raise StopIteration
		self.count += 1

		if self.return_str:
			return ' '.join(next(self.wiki_iter))
		else:
			return next(self.wiki_iter)	

class TestSecWord2Vec():

	def test_init(self):

		global wiki_keywords
		global sample_sentences

		keywords = [ 'hello', 'world' ]
		w2v = SecWord2Vec(keywords, sample_sentences, size=10, window=10, iter=1)
		assert True
		del w2v; gc.collect()

		wc  = WiKiCorpusIterator(10, return_str=True)
		w2v = SecWord2Vec(wiki_keywords, wc, size=10, window=10, iter=1)
		assert True
		del w2v; gc.collect()

	def test_train_embed(self):

		global wiki_keywords

		wc  = WiKiCorpusIterator(10, return_str=True)
		w2v = SecWord2Vec(wiki_keywords, wc, size=10, window=10, iter=1)
		w2v.train_embed()

		assert w2v.kc is not None
		assert w2v['the'] is not None
		assert w2v.kc['is'] is not None

		old_is_vec = w2v.kc['is']

		for keyword in wiki_keywords:
			assert w2v.kv[keyword].shape[0] == 10
			assert np.any(w2v.kv[keyword]) < 100

		# update with new sentences
		w2v.train_embed(
			sentences=[
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example', 'again']],
			update=True)

		assert w2v['example.'] is not None

		w2v.train_embed(
			sentences=[
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example.'],
				['This is a hello world example again']],
			update=True)

		assert not np.array_equal(old_is_vec, w2v.kv['is'])

		# TODO: Test 
		# with pytest.raises(
		# 	ValueError, match='This is a hello world example. does not exist.'):
		# 	w2v['This is a hello world example.']

		del w2v; gc.collect()


class TestSecFastTest():

	def test_init(self):

		fasttext = SecFastText()
		assert True

		wc = WiKiCorpusIterator(100)
		ft = SecFastText(wc)
		assert ft['the'] is not None
		assert ft['is'] is not None
		del ft; gc.collect()

	def test_train(self):
		pass

	# def test_train(self):

	# 	wc = WiKiCorpusIterator(100)
	# 	ft = SecFastText(wc)
	# 	ft.train(wc)

	# 	assert ft.wv['the'] is not None
	# 	assert ft.wv['is'] is not None
	# 	del ft; gc.collect()


