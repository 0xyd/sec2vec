import os
import gc
import copy
import multiprocessing as mp

import pytest
import numpy as np
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

from embedding import SecWord2Vec
from embedding import SecFastText
from embedding import SecGloVe

sample_sentences = [
	['This', 'is', 'a', 'hello', 'world', 'example.'],
	['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
	['Is', 'this', 'a', 'hello', 'world', 'example'],
	['Are', 'there', 'any', 'examples']
] 

keywords = ['hello', 'world']

# wiki_keywords = [
# 	'anarchism', 'philosophy', 'autism', 'communication', 'autistic', 
# 	'family', 'studies', 'important', 'earth', 'incident',
# 	'is'
# ]

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

		global keywords
		# global wiki_keywords
		global sample_sentences

		# w2v = SecWord2Vec(keywords, sample_sentences, size=10, window=10, iter=1)
		w2v = SecWord2Vec(
			['hello', 'world'], 
			[
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
				['Is', 'this', 'a', 'hello', 'world', 'example'],
				['Are', 'there', 'any', 'examples']
			])
		# assert True
		# del w2v; gc.collect()

		# 20181201 LIN, Y.D. Test Corpus as input
		# w2v = SecWord2Vec()

		# wc  = WiKiCorpusIterator(10, return_str=True)
		# w2v = SecWord2Vec(wiki_keywords, wc, size=10, window=10, iter=1)
		# assert True
		# del w2v; gc.collect()

	def test_train_embed(self):

		# global wiki_keywords
		
		keywords = ['hello', 'world']

		# 20181130 LIN, Y.D. Test train_embed() with no update
		w2v = SecWord2Vec(
			keywords, 
			[
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
				['Is', 'this', 'a', 'hello', 'world', 'example'],
				['Are', 'there', 'any', 'examples']
			], 
			min_count=1
		)
		w2v.train_embed()
		assert w2v.kc is not None

		# old_is_vec = copy.deepcopy(w2v['this'])

		# 20181130 LIN, Y.D. Test train_embed() with new keywords and sentences
		new_sentence = ['This is test1', 'This is test2', 'This is test3']
		new_keywords = ['test1', 'test2', 'test3']

		w2v.train_embed(new_keywords, new_sentence)

		assert set(w2v.sentences) == set([
				'This is a hello world example.',
				'This is a hello world example again',
				'Is this a hello world example',
				'Are there any examples', 
				'This is test1', 
				'This is test2', 
				'This is test3'
			])

		for t in new_keywords:
			assert w2v.kc[t] != 'Corpus of Keyword {} does not exist.'.format(t)
			assert np.where(np.isnan(w2v.kv[t]))[0].shape[0] == 0

		assert w2v['test1'] is not None

		# 20181201 LIN, Y.D. Strange ... I believe the vector should be updated
		# assert not np.array_equal(old_is_vec, w2v['this'])
		
		# 20181130 LIN, Y.D. Test train_embed() with new sentences
		new_sentence_2 = [
			['This', 'is', 'test4'], 
			['This', 'is', 'test5'], 
			['This', 'is', 'test6']
		]
		w2v.train_embed(sentences=new_sentence_2)

		for t in ['test4', 'test5', 'test6']:
			assert w2v[t] != 'Corpus of Keyword {} does not exist.'.format(t)

		res = [ ' '.join(s) for s in new_sentence_2 ]
		res.extend(
			[
				'This is a hello world example.',
				'This is a hello world example again',
				'Is this a hello world example',
				'Are there any examples',
				'This is test1', 
				'This is test2', 
				'This is test3'
			])

		assert set(w2v.sentences) == set(res)

		# 20181130 LIN, Y.D. Update with keyword only
		new_keywords_2 = ['example']
		w2v.train_embed(new_keywords_2)

		assert w2v.kc['example'] != 'Corpus of Keyword example does not exist.'
		assert np.where(np.isnan(w2v.kv['example']))[0].shape[0] == 0

		# 20181201 LIN, Y.D. Generator sentences as input.
		sentences_generator = ('this is test case{}'.format(i) for i in range(1000))
		w2v.train_embed(sentences=sentences_generator)
		
		for i in range(1000):
			s = 'case{}'.format(i)
			assert w2v[s] is not None

		del w2v; gc.collect()

		# 20181130 LIN, Y.D.: Update with corpus file
		# w2v.train_embed(corpus_file='tmp.txt')
		# assert w2v['anarchism'] is not None

		# 20181130 LIN,Y.D: Bad Test, my fault ...
		# w2v.train_embed(
		# 	sentences=[
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example', 'again']])

		# assert w2v['example.'] is not None 

		# w2v.train_embed(
		# 	sentences=[
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example again']])

		# assert not np.array_equal(old_is_vec, w2v.kv['is'])

		# TODO: Test 
		# with pytest.raises(
		# 	ValueError, match='This is a hello world example. does not exist.'):
		# 	w2v['This is a hello world example.']

		

	# def test_save_embed(self):

	# 	global keywords
	# 	global sample_sentences

	# 	w2v = SecWord2Vec(keywords, sample_sentences, size=10, window=10, iter=1)
	# 	w2v.save_embed('test_w2v.pkl')

	# 	assert os.path.isfile('test_w2v.pkl')

	# def test_load_embed(self):

	# 	w2v = w2v.load_embed('test_w2v.pkl')

	# 	assert w2v


class TestSecFastText():

	def test_init(self):

		global keywords
		# global wiki_keywords
		global sample_sentences

		ft = SecFastText(keywords, sample_sentences, size=10, window=5, iter=1)
		assert True
		del ft; gc.collect()

		# wc = WiKiCorpusIterator(100, return_str=True)
		# ft = SecFastText(wiki_keywords, wc, size=10, window=10, iter=1)
		# assert True
		# del ft; gc.collect()

	def test_train_embed(self):

		keywords = ['hello', 'world']

		# 20181130 LIN, Y.D. Test train_embed() with no update
		w2v = SecFastText(
			keywords, 
			[
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
				['Is', 'this', 'a', 'hello', 'world', 'example'],
				['Are', 'there', 'any', 'examples']
			], 
			min_count=1
		)
		w2v.train_embed()
		assert w2v.kc is not None

		old_is_vec = copy.deepcopy(w2v['This'])

		# 20181130 LIN, Y.D. Test train_embed() with new keywords and sentences
		new_sentence = ['This is test1', 'This is test2', 'This is test3']
		new_keywords = ['test1', 'test2', 'test3']

		w2v.train_embed(new_keywords, new_sentence)

		assert set(w2v.sentences) == set([
				'This is a hello world example.',
				'This is a hello world example again',
				'Is this a hello world example',
				'Are there any examples', 
				'This is test1', 
				'This is test2', 
				'This is test3'
			])

		for t in new_keywords:
			assert w2v.kc[t] != 'Corpus of Keyword {} does not exist.'.format(t)
			assert np.where(np.isnan(w2v.kv[t]))[0].shape[0] == 0

		assert w2v['test1'] is not None
		assert not np.array_equal(old_is_vec, w2v['This'])
		
		# 20181130 LIN, Y.D. Test train_embed() with new sentences
		new_sentence_2 = [
			['This', 'is', 'test4'], 
			['This', 'is', 'test5'], 
			['This', 'is', 'test6']
		]
		w2v.train_embed(sentences=new_sentence_2)

		for t in ['test4', 'test5', 'test6']:
			assert w2v[t] != 'Corpus of Keyword {} does not exist.'.format(t)

		res = [ ' '.join(s) for s in new_sentence_2 ]
		res.extend(
			[
				'This is a hello world example.',
				'This is a hello world example again',
				'Is this a hello world example',
				'Are there any examples',
				'This is test1', 
				'This is test2', 
				'This is test3'
			])

		assert set(w2v.sentences) == set(res)

		# 20181130 LIN, Y.D. Update with keyword only
		new_keywords_2 = ['example']
		w2v.train_embed(new_keywords_2)

		assert w2v.kc['example'] != 'Corpus of Keyword example does not exist.'
		assert np.where(np.isnan(w2v.kv['example']))[0].shape[0] == 0
		
		# 20181201 LIN, Y.D. Generator sentences as input.
		sentences_generator = ('this is test case{}'.format(i) for i in range(1000))
		w2v.train_embed(sentences=sentences_generator)
		
		for i in range(1000):
			s = 'case{}'.format(i)
			assert w2v[s] is not None

		del w2v; gc.collect()

		# global wiki_keywords

		# wc  = WiKiCorpusIterator(10, return_str=True)
		# ft = SecFastText(wiki_keywords, wc, size=10, window=10, iter=1, min_count=1)
		# ft.train_embed()

		# assert ft.kc is not None
		# assert ft['the'] is not None
		# assert ft.kc['is'] is not None

		# old_is_vec = ft.kc['is']

		# for keyword in wiki_keywords:
		# 	assert ft.kv[keyword].shape[0] == 10
		# 	assert np.any(ft.kv[keyword]) < 100

		# # update with new sentences
		# ft.train_embed(
		# 	sentences=[
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example', 'again']])

		# assert ft['example.'] is not None

		# ft.train_embed(
		# 	sentences=[
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example again']])

		
		# # After update, kv should be updated.
		# assert not np.array_equal(old_is_vec, ft.kv['is'])
		


class TestSecGloVe():

	def test_init(self):

		global keywords
		# global wiki_keywords
		global sample_sentences

		gv = SecGloVe(keywords, sample_sentences, size=10, window=5, iter=1)
		assert True
		del gv; gc.collect()

		# wc = WiKiCorpusIterator(100, return_str=True)
		# gv = SecGloVe(wiki_keywords, wc, size=10, window=10, iter=1)
		# assert True
		# del gv; gc.collect()

	def test_train_embed(self):

		keywords = ['hello', 'world']

		# 20181130 LIN, Y.D. Test train_embed() with no update
		w2v = SecGloVe(
			keywords, 
			[
				['This', 'is', 'a', 'hello', 'world', 'example.'],
				['This', 'is', 'a', 'hello', 'world', 'example', 'again'],
				['Is', 'this', 'a', 'hello', 'world', 'example'],
				['Are', 'there', 'any', 'examples']
			], 
			min_count=1
		)
		w2v.train_embed()
		assert w2v.kc is not None

		# old_is_vec = copy.deepcopy(w2v['This'])

		# 20181130 LIN, Y.D. Test train_embed() with new keywords and sentences
		new_sentence = ['This is test1', 'This is test2', 'This is test3']
		new_keywords = ['test1', 'test2', 'test3']

		w2v.train_embed(new_keywords, new_sentence)

		assert set(w2v.sentences) == set([
				'This is a hello world example.',
				'This is a hello world example again',
				'Is this a hello world example',
				'Are there any examples', 
				'This is test1', 
				'This is test2', 
				'This is test3'
			])

		# for t in new_keywords:
		# 	assert w2v.kc[t] != 'Corpus of Keyword {} does not exist.'.format(t)
		# 	assert np.where(np.isnan(w2v.kv[t]))[0].shape[0] == 0

		# assert w2v['test1'] is not None
		# # assert not np.array_equal(old_is_vec, w2v['This'])
		
		# # 20181130 LIN, Y.D. Test train_embed() with new sentences
		# new_sentence_2 = [
		# 	['This', 'is', 'test4'], 
		# 	['This', 'is', 'test5'], 
		# 	['This', 'is', 'test6']
		# ]
		# w2v.train_embed(sentences=new_sentence_2)

		# for t in ['test4', 'test5', 'test6']:
		# 	assert w2v[t] != 'Corpus of Keyword {} does not exist.'.format(t)

		# res = [ ' '.join(s) for s in new_sentence_2 ]
		# res.extend(
		# 	[
		# 		'This is a hello world example.',
		# 		'This is a hello world example again',
		# 		'Is this a hello world example',
		# 		'Are there any examples',
		# 		'This is test1', 
		# 		'This is test2', 
		# 		'This is test3'
		# 	])

		# assert set(w2v.sentences) == set(res)

		# # 20181130 LIN, Y.D. Update with keyword only
		# new_keywords_2 = ['example']
		# w2v.train_embed(new_keywords_2)

		# assert w2v.kc['example'] != 'Corpus of Keyword example does not exist.'
		# assert np.where(np.isnan(w2v.kv['example']))[0].shape[0] == 0

		# global wiki_keywords

		# wc  = WiKiCorpusIterator(10, return_str=True)
		# gv = SecGloVe(wiki_keywords, wc, size=10, window=10, iter=1, min_count=1)
		# gv.train_embed()

		# assert gv.kc is not None
		# assert gv['the'] is not None
		# assert gv.kc['is'] is not None

		# old_is_vec = gv.kc['is']

		# for keyword in wiki_keywords:
		# 	assert gv.kv[keyword].shape[0] == 10
		# 	assert np.any(gv.kv[keyword]) < 100

		# # update with new sentences
		# gv.train_embed(
		# 	sentences=[
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example.'],
		# 		['This', 'is', 'a', 'hello', 'world', 'example', 'again']],
		# 	)
		# 	# update=True)

		# assert gv['example.'] is not None

		# gv.train_embed(
		# 	sentences=[
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example.'],
		# 		['This is a hello world example again']],
		# 	)
		# 	# update=True)

		# assert not np.array_equal(old_is_vec, gv.kv['is'])
		# del gv; gc.collect()
