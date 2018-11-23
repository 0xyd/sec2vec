# -*- coding: utf-8 -*-
# import re
from multiprocessing import Pool, cpu_count

from flashtext import KeywordProcessor


def mp_extract_keywords(
	keywords, sentences, case_sensitive=False):

	corpus = dict()
	kp = KeywordProcessor(case_sensitive=case_sensitive)

	for keyword in keywords:

		corpus[keyword] = []
		kp.add_keyword(keyword, ' ')

		for sentence in sentences:

			keywords_found = kp.extract_keywords(sentence)

			if keywords_found:

				tokens = list(filter(
					lambda s: s if len(s) > 0 else None, 
					kp.replace_keywords(sentence).split(' ')))
				corpus[keyword].append(tokens)

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

	def __init__(self, keywords, case_sensitive=False):

		self.keyword_corpus = KeywordCorpus()
		self.case_sensitive = case_sensitive

		for keyword in keywords:
			self.keyword_corpus[keyword] = []

	def create(self, sentences, chunksize=256, corpus_worker=3):

		sentences_chunk = []
		partition_size = chunksize // corpus_worker
		keywords = list(self.keyword_corpus.keys())
		corpus_pool = Pool(corpus_worker)

		for i, sentence in enumerate(sentences):

			if i % (chunksize-1) == 0 and i > 0:

				partitions = []

				for i in range(corpus_worker):

					if i == corpus_worker-1:
						partitions.append(
							sentences_chunk[i*partition_size:])
					else:
						partitions.append(
							sentences_chunk[i*partition_size:(i+1)*partition_size])

				new_corpus_list = corpus_pool.starmap(
					mp_extract_keywords, 
					((keywords, partition, self.case_sensitive) for partition in partitions))

				for new_corpus in new_corpus_list:
					for keyword, tokens in new_corpus.items():
						self.keyword_corpus[keyword].append(tokens)

				sentences_chunk = []

			else:
				sentences_chunk.append(sentence)

		corpus_pool.close()

		if sentences_chunk:

			new_corpus = mp_extract_keywords(keywords, sentences_chunk)
			for keyword, tokens in new_corpus.items():
				self.keyword_corpus[keyword].append(tokens)

		return self.keyword_corpus


	def update_keywords(self, keywords):

		for keyword in keywords:
			if keyword in self.keyword_corpus:
				raise ValueError("Keyword {} is exist already".format(keyword))
			else:
				self.keyword_corpus[keyword] = []

	def update(self, sentences, chunksize=256, corpus_worker=3):
		
		pass


