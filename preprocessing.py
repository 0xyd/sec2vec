# -*- coding: utf-8 -*-
# import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from flashtext import KeywordProcessor


class SentenceIterator():

	def __init__(self, sentences): 
		self.iterable = (s for s in sentences)

		# 20181130 Hannah Chen, add length for iterable
		self.length = sum(1 for _ in sentences)

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

	# 20181130 Hannah Chen, add length for iterable
	def __len__(self):
		return self.length

# def mp_extract_keywords(
# 	keywords, sentences, case_sensitive=False):

# 	corpus = dict()
# 	kp = KeywordProcessor(case_sensitive=case_sensitive)

# 	for keyword in keywords:

# 		corpus[keyword] = set()
# 		kp.add_keyword(keyword, ' ')

# 		for sentence in sentences:

# 			if isinstance(sentence, list):
# 				sentence = ' '.join(sentence)

# 			# 20181123 LIN, Y.D. Remove Duplicates
# 			if sentence in corpus[keyword]:
# 				continue

# 			keywords_found = kp.extract_keywords(sentence)

# 			if keywords_found:

# 				# 20181123 LIN, Y.D. Reserved keywords.
# 				corpus[keyword].add(sentence)

# 				# tokens = list(filter(
# 				# 	lambda s: s if len(s) > 0 else None, sentence.split(' ')))

# 				# tokens = list(filter(
# 				# 	lambda s: s if len(s) > 0 else None, 
# 				# 	kp.replace_keywords(sentence).split(' ')))
# 				# corpus[keyword].append(tokens)

# 		kp.remove_keyword(keyword)

# 	return corpus

# 20181128 Hannah Chen, Optimize performance
def mp_extract_keywords(keywords, sentences, case_sensitive=False):
	
	corpus = dict()
	kp = KeywordProcessor(case_sensitive=case_sensitive)
	
	kp.add_keywords_from_list(keywords)

	for sentence in sentences:
		
		if isinstance(sentence, list):
			sentence = ' '.join(sentence)

		keywords_found = kp.extract_keywords(sentence)

		for keyword in keywords_found:
			corpus.setdefault(keyword, set())
			corpus[keyword].add(sentence)

	return corpus


class KeywordCorpus(dict):
	
	def __setitem__(self, keyword, corpus):
		super().__setitem__(keyword, corpus)

	def __getitem__(self, keyword):
		return super().get(keyword, f'Corpus of Keyword {keyword} does not exist.')


class KeywordCorpusIterator():

	def __init__(self, keyword_corpus, return_tokens=True):

		if return_tokens:
			self.iterable = (
				sentence.split(' ') for corpus in keyword_corpus.values() for sentence in corpus)
		else:
			self.iterable = (
				sentence for corpus in keyword_corpus.values() for sentence in corpus)

	def __iter__(self): return self

	def __next__(self):
		try:
			return next(self.iterable)
		except:
			raise StopIteration

class KeywordCorpusFactory():

	def __init__(self, keywords, case_sensitive=False, worker=3):

		self.kc = KeywordCorpus()
		self.case_sensitive = case_sensitive
		self.corpus_worker = worker

		for keyword in keywords:
			self.kc[keyword] = set()
			# self.kc[keyword] = []

	def _create(self, keywords, sentences, chunksize=5000):

		sentences_chunk = []
		partition_size = chunksize // self.corpus_worker
		corpus_pool = Pool(self.corpus_worker)

		for i, sentence in tqdm(enumerate(sentences), total=sentences.__len__()):

			if i % (chunksize-1) == 0 and i > 0:

				partitions = []

				for i in range(self.corpus_worker):

					if i == self.corpus_worker-1:
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
						self.kc[keyword] = \
							self.kc[keyword].union(sentences)

					# for keyword, tokens in new_corpus.items():
					# 	self.kc[keyword].extend(tokens)
						# self.kc[keyword].append(tokens)

				sentences_chunk = []

			else:
				sentences_chunk.append(sentence)

		corpus_pool.close()

		if sentences_chunk:

			new_corpus = mp_extract_keywords(keywords, sentences_chunk)
			for keyword, sentences in new_corpus.items():
				self.kc[keyword] = \
					self.kc[keyword].union(sentences)

				# self.kc[keyword].extend(tokens)
				# self.kc[keyword].append(tokens)

	def create(self, sentences, chunksize=5000):

		keywords = list(self.kc.keys())

		# 20181130 Hannah Chen, create with sentence iterator
		self._create(keywords, sentences, chunksize=chunksize)
		# self._create(keywords, sentences, chunksize=256)

		# 20181129 Hannah Chen, return error if keyword corpus is empty
		# if all(len(value) == 0 for value in self.kc.values()):
		# 	raise Exception("No keywords found in input sentences")

		return self.kc

		# for i, sentence in enumerate(sentences):

		# 	if i % (chunksize-1) == 0 and i > 0:

		# 		partitions = []

		# 		for i in range(self.corpus_worker):

		# 			if i == self.corpus_worker-1:
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
		# 				self.kc[keyword].append(tokens)

		# 		sentences_chunk = []

		# 	else:
		# 		sentences_chunk.append(sentence)

		# corpus_pool.close()

		# if sentences_chunk:

		# 	new_corpus = mp_extract_keywords(keywords, sentences_chunk)
		# 	for keyword, tokens in new_corpus.items():
		# 		self.kc[keyword].append(tokens)

		# return self.kc


	# def update_keywords(self, keywords):

	# 	for keyword in keywords:
	# 		if keyword in self.kc:
	# 			raise ValueError("Keyword {} is exist already".format(keyword))
	# 		else:
	# 			self.kc[keyword] = []

	def update(self, keywords=None, sentences=None, chunksize=5000):

		if keywords is None and sentences is None:
			raise ValueError(
				'One of parameters between keywords and sentences should not be None')
		if sentences:
			self.create(sentences, chunksize)

		if keywords:
			for keyword in keywords:
				if keyword in self.kc:
					raise ValueError("Keyword {} is exist already".format(keyword))
				else:
					self.kc[keyword] = set()

			# Retrieve old tokens and assemble them to sentences.
			sentences_gen = (
				sentences for corpus in self.kc.values() 
					for sentences in corpus)
			
			self._create(keywords, sentences_gen, chunksize)
