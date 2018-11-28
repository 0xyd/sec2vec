import logging
import subprocess
from subprocess import Popen
from subprocess import PIPE
from collections import Iterator
from multiprocessing import cpu_count

import tqdm
import numpy as np
#from glove import Glove, Corpus
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec

from logger import EpochLogger
from preprocessing import KeywordCorpusFactory
from preprocessing import KeywordCorpusIterator


epoch_logger = EpochLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

			self.build_vocab(SentenceIterator(sentences), update=update)
			self.update(keywords, SentenceIterator(sentences))
			self.train(
				SentenceIterator(sentences), corpus_file, 
				total_examples, total_words, epochs, 
				start_alpha, end_alpha, word_count, 
				queue_factor, report_delay, compute_loss)
		else:

			self.train(
				KeywordCorpusIterator(self.kc), 
				corpus_file, total_examples, total_words, epochs, 
				start_alpha, end_alpha, word_count, 
				queue_factor, report_delay, compute_loss)

			self.wv['unk'] = np.random.uniform(-1, 1, (self.vector_size,))

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
		self, keywords, sentences, 
		corpus_worker, corpus_chunksize, case_sensitive,
		window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=cpu_count(), 
		min_alpha=0.0001, sg=0, hs=0, 
		negative=5, ns_exponent=0.75, cbow_mean=1, 
		iter=5, null_word=0, trim_rule=None, 
		sorted_vocab=1, batch_words=10000, compute_loss=False, 
		max_final_vocab=None):

		KeywordCorpusFactory.__init__(self, keywords, case_sensitive)
		self.kc = self.create(sentences, corpus_chunksize, corpus_worker)
		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize

		FastText.__init__(self, 
			window, min_count, max_vocab_size, 
			sample, seed, workers, 
			min_alpha, sg, hs, 
			negative, ns_exponent, cbow_mean, 
			iter, null_word, trim_rule, 
			sorted_vocab, batch_words, compute_loss, 
			max_final_vocab)


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
		sorted_vocab=1, batch_words=10000, compute_loss=False, 
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
		self, sentences=None, corpus_file=None, 
		sg=0, hs=0, size=100, alpha=0.025, 
		window=5, min_count=5, max_vocab_size=None,
		word_ngrams=1, sample=0.001, seed=1, 
		workers=3, min_alpha=0.0001, negative=5, 
		ns_exponent=0.75, cbow_mean=1, iter=5, 
		null_word=0, min_n=3, max_n=6, sorted_vocab=1, 
		bucket=2000000, trim_rule=None, batch_words=10000):


		super().__init__(
			sentences=sentences, corpus_file=corpus_file, 
			sg=sg, hs=hs, size=size, alpha=alpha, 
			window=window, min_count=min_count, max_vocab_size=max_vocab_size,
			word_ngrams=word_ngrams, sample=sample, seed=seed, 
			workers=workers, min_alpha=min_alpha, negative=negative, 
			ns_exponent=ns_exponent, cbow_mean=cbow_mean, iter=iter, 
			null_word=null_word, min_n=min_n, max_n=max_n, sorted_vocab=sorted_vocab, 
			bucket=bucket, trim_rule=trim_rule, batch_words=batch_words)

		self.build_vocab(
			(corpus for corpus in KeywordCorpusIterator(self.kc)))



class KeywordCorpusFactoryGloveMixin(Sec2Vec, KeywordCorpusFactory):

	def __init__(
		self, keywords, sentences, corpus_file, 
		corpus_worker, corpus_chunksize, case_sensitive,
		vocab_file, save_file, 
		size, window, min_count, threads,
		iters, X_max, memory,
		update, pre_train_model, new_model_name
		):

		KeywordCorpusFactory.__init__(self, keywords, case_sensitive)
		self.kc = self.create(sentences, corpus_chunksize)
		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize    
		



#11/24 add 
class SecGloVe(KeywordCorpusFactoryGloveMixin):

	

	def __init__(
		self, keywords, sentences=None,
		corpus_file=None, 
		corpus_worker=3, corpus_chunksize=256, case_sensitive=False, 
		vocab_file='vocab.txt', save_file='vector',
		min_count=5, size=100, window=5, threads=3,
		iters=5, X_max=10, memory=4.0, update=False,
		pre_train_model=None, new_model_name=None
		):

		super().__init__(
			keywords, sentences,
			corpus_file, corpus_worker,
			corpus_chunksize, case_sensitive,
			vocab_file, save_file,
			min_count, size, window, threads,
			iters, X_max, memory, update,
			pre_train_model, new_model_name)

		self.keywords = keywords
		self.sentences = sentences
		self.corpus_file = corpus_file
		self.corpus_worker = corpus_worker
		self.corpus_chunksize = corpus_chunksize
		self.case_sensitive = case_sensitive
		self.vocab_file = vocab_file
		self.save_file = save_file
		self.min_count = min_count
		self.size = size
		self.window = window
		self.threads = threads
		self.iters = iters
		self.X_max = X_max
		self.memory = memory
		self.update = update
		self.pre_train_model = pre_train_model
		self.new_model_name = new_model_name

		assert self.sentences or self.corpus_file

		if self.sentences and not self.corpus_file:
			sentences = (corpus for corpus in KeywordCorpusIterator(self.kc))
			f = open('./glove/temp_glove_sentence.txt', 'w+')

			for sentence in sentences:
				f.write(' '.join(sentence))
				f.write('\n')

			self.corpus_file = 'temp_glove_sentence.txt'

			f.close()



	def train_glove_embed(self):

		if self.update and self.pre_train_model:  #是否需要update

			if isinstance(self.sentences, Iterator):
				raise ValueError(
					'sentences accpets list of str or list of tokens only.')

			glove2word2vec(glove_input_file=self.pre_train_model, \
						   word2vec_output_file="./glove/glove_vectors_gensim.vec")

			pre_train_model = "./glove/glove_vectors_gensim.vec"
			pre_trained_vec = KeyedVectors.load_word2vec_format(pre_train_model, binary=False)
			
			new_model = Word2Vec(size=self.size, min_count=self.min_count)
			new_model.build_vocab(sentences)
			total_examples = new_model.corpus_count

			new_model.build_vocab([list(self.pretrained_vec.vocab.keys())], update=self.update)
			new_model.intersect_word2vec_format(pre_train_model, binary=False, lockf=1.0)
			new_model.train(sentences, total_examples=total_examples, epochs=self.iters)
			new_model.save(self.new_model_name + '.bin')


		else:

			argument = [
				'./demo_v2.sh', '--Corpus_File={}'.format(self.corpus_file),
				'--Save_File={}'.format(self.save_file), '--Vocab_File={}'.format(self.vocab_file),
				'--Vocab_Min_Count={}'.format(self.min_count), '--Vector_Size={}'.format(self.size),
				'--Window={}'.format(self.window), '--Threads={}'.format(self.threads),
				'--iters={}'.format(self.iters), '--X_max={}'.format(self.X_max),
				'--Memory={}'.format(self.memory)
			]
			process = subprocess.Popen(argument, stdin=PIPE, stdout=PIPE, cwd='glove/')
			
			for line in process.stdout:
				logging.info(line.decode('utf-8').strip())

	def remove_temp_file(self):
		if self.corpus_file == 'temp_glove_sentence.txt':
			subprocess.run(['rm','-rf',self.corpus_file],cwd='glove/')

		



