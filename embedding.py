from collections import Iterator
from multiprocessing import cpu_count

import numpy as np
from glove import Glove, Corpus
from gensim.models import Word2Vec, FastText

from logger import EpochLogger
from preprocessing import KeywordCorpusFactory
from preprocessing import KeywordCorpusIterator


# epoch_logger = EpochLogger()

class SentenceIterator():

	def __init__(self, sentences): self.iterable = (s for s in sentences)

	def __iter__(self): return self

	def __next__(self):

		s = next(self.iterable)

		if isinstance(s, list):
			return s
		elif isinstance(s, str):
			return s.split(' ')
		else:
			return ValueError(
				'Only String or list of string are acceptable.')

class Sec2Vec():

	def __init__(self): pass

	def __getitem__(self, word):

		try:
			return self.wv[word]
		except:
			return ValueError('{} does not exist.'.format(word))

	def _get_vec(self, token):

		if token in self.wv:
			return self.wv[token]
		else:
			return self.wv['unk']

	def _cal_kv(self):

		for keyword, sentences in self.kc.items():

			kv = np.zeros((self.vector_size, ))
			token_count = self.keyword_count[keyword] if self.keyword_count[keyword] else 0

			for sentence in sentences:

				for token in sentence.split(' '):

					if token == keyword:
						continue

					if token_count: 
						kv = kv + self._get_vec(token)
					else:
						kv = self._get_vec(token)

					token_count += 1

				kv = kv / token_count

				self.kv[keyword] = kv
				self.keyword_count[keyword] += token_count

	def train_embed(
		self, keywords=None, sentences=None, corpus_file=None, update=False,
		total_examples=None, total_words=None,  epochs=None, 
		start_alpha=None, end_alpha=None, word_count=0, 
		queue_factor=2, report_delay=1.0, compute_loss=False):


		epochs = epochs if epochs else self.epochs
		total_examples = total_examples if total_examples else self.corpus_count

		if update:

			if isinstance(sentences, Iterator):
				raise ValueError(
					'sentences accpets list of str or list of tokens only.')

			print('sentences: {}'.format(sentences))
			print('update: {}'.format(update))
			print('keywords: {}'.format(keywords))

			self.build_vocab(SentenceIterator(sentences), update=update)
			self.update(keywords, SentenceIterator(sentences))

			print('Number of vocabs: {}'.format(len(self.wv.vocab)))
			print('Last word: {}'.format(list(self.wv.vocab.keys())[-1]))

			# 20181126 Hannah Chen, check compute_loss 
			#(FastText does not contain this variable)
			if compute_loss:
				self.train(
					SentenceIterator(sentences), corpus_file, 
					total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count, 
					queue_factor, report_delay, compute_loss)
			else:
				self.train(
					SentenceIterator(sentences), corpus_file, 
					total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count, 
					queue_factor, report_delay)
		else:

			# 20181127 Hannah Chen, assign 'unk' before training 
			self.wv['unk'] = np.random.uniform(-1, 1, (self.vector_size,))

			# 20181126 Hannah Chen, check compute_loss 
			#(FastText does not contain this variable)
			if compute_loss:

				self.train(
					KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count, compute_loss,
					queue_factor, report_delay)
			else:

				# 20181127 Hannah Chen, append word vector of 'unk' 
				# to the array that collects all word vectors
				self.wv.vectors_vocab = np.vstack((self.wv.vectors_vocab, self.wv['unk']))

				self.train(
					KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count,
					queue_factor, report_delay)

			# self.wv['unk'] = np.random.uniform(-1, 1, (self.vector_size,))


		self._cal_kv()



class KeywordCorpusFactoryWord2VecMixin(Sec2Vec, Word2Vec, KeywordCorpusFactory): 

	def __init__(
		self, keywords, sentences, 
		corpus_worker, corpus_chunksize, case_sensitive, 
		corpus_file, size, alpha, 
		window, min_count, max_vocab_size, 
		sample, seed, workers, 
		min_alpha, sg, hs, 
		negative, ns_exponent, cbow_mean, 
		iter, null_word, trim_rule, 
		sorted_vocab, batch_words, compute_loss, 
		max_final_vocab):
		
		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)
		self.kc = self.create(sentences, corpus_chunksize)
		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize

		# 20181126 Hannah Chen, initialize epoch_logger
		epoch_logger = EpochLogger(compute_loss)
		
		Word2Vec.__init__(
			self, 
			corpus_file=corpus_file, size=size, 
			alpha=alpha, window=window, min_count=min_count,
			max_vocab_size=max_vocab_size, sample=sample, seed=seed, 
			workers=workers, min_alpha=min_alpha, sg=sg, 
			hs=hs, negative=negative, ns_exponent=ns_exponent, 
			cbow_mean=cbow_mean, iter=iter, null_word=null_word, 
			trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words, 
			compute_loss=compute_loss, max_final_vocab=max_final_vocab,
			callbacks=[epoch_logger])

		
class KeywordCorpusFactoryFasttextMixin(Sec2Vec, FastText, KeywordCorpusFactory):

	def __init__(
		# self, keywords, sentences, 
		# corpus_worker, corpus_chunksize, case_sensitive,
		# window=5, min_count=5, max_vocab_size=None, 
		# sample=0.001, seed=1, workers=cpu_count(), 
		# min_alpha=0.0001, sg=0, hs=0, 
		# negative=5, ns_exponent=0.75, cbow_mean=1, 
		# iter=5, null_word=0, trim_rule=None, 
		# sorted_vocab=1, batch_words=10000, compute_loss=False, 
		# max_final_vocab=None

		# 20181126 Hannah Chen, modified variables 
		self, keywords, sentences, corpus_file,
		size, alpha, word_ngrams, min_n, max_n, bucket,
		corpus_worker, corpus_chunksize, case_sensitive,
		window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=3, 
		min_alpha=0.0001, sg=0, hs=0, compute_loss=False,
		negative=5, ns_exponent=0.75, cbow_mean=1, 
		iter=5, null_word=0, trim_rule=None, 
		sorted_vocab=1, batch_words=10000):

		# 20181126 Hannah Chen, modified variable: corpus_worker
		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)
		self.kc = self.create(sentences, corpus_chunksize)
		# self.kc = self.create(sentences, corpus_chunksize, corpus_worker)

		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.compute_loss = compute_loss

		# 20181126 Hannah Chen, Add keyword_count
		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))

		# self.corpus_worker = corpus_worker
		self.corpus_chunksize = corpus_chunksize

		# 20181126 Hannah Chen, initialize epoch_logger
		epoch_logger = EpochLogger(compute_loss)

		FastText.__init__(self, 
			# window, min_count, max_vocab_size, 
			# sample, seed, workers, 
			# min_alpha, sg, hs, 
			# negative, ns_exponent, cbow_mean, 
			# iter, null_word, trim_rule, 
			# sorted_vocab, batch_words, compute_loss, 
			# max_final_vocab

			# 20181126 Hannah Chen, modified variables 
			corpus_file=corpus_file, size=size, alpha=alpha, word_ngrams=word_ngrams,
			window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
			sample=sample, seed=seed, workers=workers, min_n=min_n, max_n=max_n,
			min_alpha=min_alpha, sg=sg, hs=hs, bucket=bucket,
			negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean, 
			iter=iter, null_word=null_word, trim_rule=trim_rule, 
			sorted_vocab=sorted_vocab, batch_words=batch_words, 
			callbacks=[epoch_logger])


class SecWord2Vec(KeywordCorpusFactoryWord2VecMixin):

	def __init__(
		self, keywords, sentences, 
		corpus_worker=3, corpus_chunksize=256, case_sensitive=False, 
		corpus_file=None, size=100, alpha=0.025, 
		window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=cpu_count(), 
		min_alpha=0.0001, sg=0, hs=0, 
		negative=5, ns_exponent=0.75, cbow_mean=1, 
		iter=5, null_word=0, trim_rule=None, 
		sorted_vocab=1, batch_words=10000, compute_loss=True, 
		max_final_vocab=None):
		
		super().__init__( 
			keywords, sentences, corpus_worker, 
			corpus_chunksize, case_sensitive, corpus_file, 
			size, alpha, window, 
			min_count, max_vocab_size, sample, 
			seed, workers, min_alpha, 
			sg, hs, negative, 
			ns_exponent, cbow_mean, iter, 
			null_word, trim_rule, sorted_vocab,
			batch_words, compute_loss, max_final_vocab)

		self.build_vocab(
			(corpus for corpus in KeywordCorpusIterator(self.kc)))

class SecFastText(KeywordCorpusFactoryFasttextMixin):

	def __init__(
		# self, sentences=None, corpus_file=None, 
		# sg=0, hs=0, size=100, alpha=0.025, 
		# window=5, min_count=5, max_vocab_size=None,
		# word_ngrams=1, sample=0.001, seed=1, 
		# workers=3, min_alpha=0.0001, negative=5, 
		# ns_exponent=0.75, cbow_mean=1, iter=5, 
		# null_word=0, min_n=3, max_n=6, sorted_vocab=1, 
		# bucket=2000000, trim_rule=None, batch_words=10000

		# 20181126 Hannah Chen, modified variables 
		self, keywords, sentences, 
		corpus_worker=3, corpus_chunksize=256, case_sensitive=False,
		corpus_file=None, size=100, alpha=0.025,
		window=5, min_count=5, max_vocab_size=None,
		sample=0.001, seed=1, workers=3, compute_loss=False,
		min_alpha=0.0001, sg=0, hs=0, word_ngrams=1, 
		negative=5, ns_exponent=0.75, cbow_mean=1, 
		iter=5, null_word=0, min_n=3, max_n=6, sorted_vocab=1, 
		bucket=2000000, trim_rule=None, batch_words=10000):


		super().__init__(
			# sentences=sentences, corpus_file=corpus_file, 
			# sg=sg, hs=hs, size=size, alpha=alpha, 
			# window=window, min_count=min_count, max_vocab_size=max_vocab_size,
			# word_ngrams=word_ngrams, sample=sample, seed=seed, 
			# workers=workers, min_alpha=min_alpha, negative=negative, 
			# ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, 
			# null_word=null_word, min_n=min_n, max_n=max_n, sorted_vocab=sorted_vocab, 
			# bucket=bucket, trim_rule=trim_rule, batch_words=batch_words

			# 20181126 Hannah Chen, modified variables 
			keywords=keywords, sentences=sentences, corpus_file=corpus_file, 
			sg=sg, hs=hs, size=size, alpha=alpha, corpus_worker=corpus_worker, 
			case_sensitive=case_sensitive, corpus_chunksize=corpus_chunksize,
			window=window, min_count=min_count, max_vocab_size=max_vocab_size,
			word_ngrams=word_ngrams, sample=sample, seed=seed,
			workers=workers, min_alpha=min_alpha, negative=negative, 
			ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, 
			null_word=null_word, min_n=min_n, max_n=max_n, sorted_vocab=sorted_vocab, 
			bucket=bucket, trim_rule=trim_rule, batch_words=batch_words)

		self.build_vocab(
			(corpus for corpus in KeywordCorpusIterator(self.kc)))


class SecGloVe(Glove):

	pass


