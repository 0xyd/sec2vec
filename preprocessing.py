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

class KeywordCorpusFactory():
# class KeywordCorpusFactory(KeywordProcessor):

	def __init__(self, keywords, case_sensitive=False):

		# super().__init__(case_sensitive=case_sensitive)
		self.keyword_corpus = KeywordCorpus()
		self.case_sensitive = case_sensitive

		for keyword in keywords:
			# self.add_keyword(keyword, ' ')
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

		# self.keyword_corpus[keyword] = list(
		# 	list(
		# 		filter(
		# 			lambda s: s if len(s) > 0 else None, 
		# 			self.replace_keywords(sentence).split(' '))) for sentence in sentences)


	# def create_keyword_corpus(self, keyword, sentences):

	#     self.add_keyword(keyword, ' ')

	#     self.keyword_corpus[keyword] = list(
	#         list(
	#             filter(
	#                 lambda s: s if len(s) > 0 else None, 
	#                 self.replace_keywords(sentence).split(' '))) for sentence in sentences)


	# def create_keyword_corpus_from_file(self, keyword, file_path):
		
	# 	pass
		

# def clean_keyword_in_sentence(keyword, sentence):
# 	'''
# 	substitute keyword in sentence to ''
	
# 	:param keyword: 
# 	:type keyword: str
	
# 	:param sentence: 
# 	:type sentence: str
	
# 	'''
# 	return re.sub(keyword, '', sentence , flags=re.I)



# def write_sentence_in_dict(keyword_dict, keyword, clean_sentence):
# 	'''

# 	write preprocessed sentence in keyword_dict , ex: {cve_id : [sentence] }, if sentence duplicate, not append 
	
# 	:param keyword_dict: the keyword dict
# 	:type keyword_dict: dict

# 	:param keyword: 
# 	:type keyword: str
	
# 	:param clean_sentence: the preprocessed sentence   
# 	:type clean_sentence: str
	
# 	'''
	
# 	clean_sentence = clean_keyword_in_sentence(keyword, clean_sentence)
# 	#add all cve corpus to dict
# 	if keyword not in keyword_dict:
# 		keyword_dict[keyword] = [clean_sentence]
# 	else:
# 		#detect duplicate data
# 		if clean_sentence not in keyword_dict[keyword]:
# 			keyword_dict[keyword].append(clean_sentence)
