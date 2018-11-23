# -*- coding: utf-8 -*-
# import re
from multiprocessing import Pool, cpu_count

from flashtext import KeywordProcessor


def mp_extract_keywords(
	keywords, sentences, case_sensitive=False):

	corpus = dict()
	kp = KeywordProcessor(case_sensitive=case_sensitive)

	for keyword in keywords:

		corpus[keyword] = set()
		kp.add_keyword(keyword, ' ')

		for sentence in sentences:

			if isinstance(sentence, list):
				sentence = ' '.join(sentence)

			# 20181123 LIN, Y.D. Remove Duplicates
			if sentence in corpus[keyword]:
				continue

			keywords_found = kp.extract_keywords(sentence)

			if keywords_found:

				# 20181123 LIN, Y.D. Reserved keywords.
				corpus[keyword].add(sentence)

				# tokens = list(filter(
				# 	lambda s: s if len(s) > 0 else None, sentence.split(' ')))

				# tokens = list(filter(
				# 	lambda s: s if len(s) > 0 else None, 
				# 	kp.replace_keywords(sentence).split(' ')))
				# corpus[keyword].append(tokens)

		kp.remove_keyword(keyword)

	return corpus


class KeywordCorpus(dict):
	
	def __setitem__(self, keyword, corpus):
		super().__setitem__(keyword, corpus)

	def __getitem__(self, keyword):
		return super().get(keyword, f'Corpus of Keyword {keyword} does not exist.')


class KeywordCorpusIterator():

	def __init__(self, keyword_corpus):
		self.iterable = (tokens for corpus in  keyword_corpus.values for tokens in corpus)

	def __iter__(self): return self

	def __next__(self):
		try:
			return next(self.iterable)
		except:
			raise StopIteration

class KeywordCorpusFactory():

	def __init__(self, keywords, case_sensitive=False, worker=3):

		self.keyword_corpus = KeywordCorpus()
		self.case_sensitive = case_sensitive
		self.worker = worker

		for keyword in keywords:
			self.keyword_corpus[keyword] = set()
			# self.keyword_corpus[keyword] = []

	def _create(self, keywords, sentences, chunksize=256):

		sentences_chunk = []
		partition_size = chunksize // self.worker
		corpus_pool = Pool(self.worker)

		for i, sentence in enumerate(sentences):

			if i % (chunksize-1) == 0 and i > 0:

				partitions = []

				for i in range(self.worker):

					if i == self.worker-1:
						partitions.append(
							sentences_chunk[i*partition_size:])
					else:
						partitions.append(
							sentences_chunk[i*partition_size:(i+1)*partition_size])

				new_corpus_list = corpus_pool.starmap(
					mp_extract_keywords, 
					((keywords, partition, self.case_sensitive) for partition in partitions))

				for new_corpus in new_corpus_list:
					for keyword, sentences in new_corpus.items():
						self.keyword_corpus[keyword] = \
							self.keyword_corpus[keyword].union(sentences)

					# for keyword, tokens in new_corpus.items():
					# 	self.keyword_corpus[keyword].extend(tokens)
						# self.keyword_corpus[keyword].append(tokens)

				sentences_chunk = []

			else:
				sentences_chunk.append(sentence)

		corpus_pool.close()

		if sentences_chunk:

			new_corpus = mp_extract_keywords(keywords, sentences_chunk)
			for keyword, sentences in new_corpus.items():
				self.keyword_corpus[keyword] = \
					self.keyword_corpus[keyword].union(sentences)

				# self.keyword_corpus[keyword].extend(tokens)
				# self.keyword_corpus[keyword].append(tokens)

	def create(self, sentences, chunksize=256):

		keywords = list(self.keyword_corpus.keys())
		self._create(keywords, sentences, chunksize=256)
		return self.keyword_corpus
		
		# for i, sentence in enumerate(sentences):

		# 	if i % (chunksize-1) == 0 and i > 0:

		# 		partitions = []

		# 		for i in range(self.worker):

		# 			if i == self.worker-1:
		# 				partitions.append(
		# 					sentences_chunk[i*partition_size:])
		# 			else:
		# 				partitions.append(
		# 					sentences_chunk[i*partition_size:(i+1)*partition_size])

		# 		new_corpus_list = corpus_pool.starmap(
		# 			mp_extract_keywords, 
		# 			((keywords, partition, self.case_sensitive) for partition in partitions))

		# 		for new_corpus in new_corpus_list:
		# 			for keyword, tokens in new_corpus.items():
		# 				self.keyword_corpus[keyword].append(tokens)

		# 		sentences_chunk = []

		# 	else:
		# 		sentences_chunk.append(sentence)

		# corpus_pool.close()

		# if sentences_chunk:

		# 	new_corpus = mp_extract_keywords(keywords, sentences_chunk)
		# 	for keyword, tokens in new_corpus.items():
		# 		self.keyword_corpus[keyword].append(tokens)

		# return self.keyword_corpus


	# def update_keywords(self, keywords):

	# 	for keyword in keywords:
	# 		if keyword in self.keyword_corpus:
	# 			raise ValueError("Keyword {} is exist already".format(keyword))
	# 		else:
	# 			self.keyword_corpus[keyword] = []

	def update(self, keywords=None, sentences=None, chunksize=256):

		if keywords is None and sentences is None:
			raise ValueError(
				'One of parameters between keywords and sentences should not be None')
		if sentences:
			self.create(sentences, chunksize)

		if keywords:
			for keyword in keywords:
				if keyword in self.keyword_corpus:
					raise ValueError("Keyword {} is exist already".format(keyword))
				else:
					self.keyword_corpus[keyword] = set()

			# Retrieve old tokens and assemble them to sentences.
			sentences_gen = (
				sentences for corpus in self.keyword_corpus.values() 
					for sentences in corpus)
			
			self._create(keywords, sentences_gen, chunksize)
