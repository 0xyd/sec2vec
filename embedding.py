import logging
import subprocess
from subprocess import Popen
from subprocess import PIPE
from collections import Iterator
from multiprocessing import cpu_count

# import tqdm
import numpy as np
#from glove import Glove, Corpus
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec

from logger import EpochLogger
from preprocessing import KeywordCorpusFactory
from preprocessing import KeywordCorpusIterator


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

		# if hasattr(self, 'wv'):

		if token in self.wv:
			return self.wv[token]
		else:
			return self.wv['<unk>']

		# else:

		# 	if token in self.kv:
		# 		return self.kv[token]
		# 	else:
		# 		return self.kv['<unk>']

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


			# 20181126 Hannah Chen, check compute_loss 
			#(FastText does not contain this variable)
			if compute_loss:

				self.train(
					KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count, compute_loss,
					queue_factor, report_delay)

			else:

				# # 20181127 Hannah Chen, append word vector of 'unk' 
				# # to the array that collects all word vectors
				# self.wv.vectors_vocab = np.vstack((self.wv.vectors_vocab, self.wv['unk']))

				self.train(
					KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count,
					queue_factor, report_delay)

			self.wv['<unk>'] = np.random.uniform(-1, 1, (self.vector_size,))

			# 20181127 Hannah Chen, append word vector of 'unk' 
			# to the array that collects all word vectors
			if compute_loss:
				self.wv.vectors_vocab = np.vstack((self.wv.vectors_vocab, self.wv['<unk>']))


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
		self, keywords, sentences, corpus_file=None,
		size=100, alpha=0.025, word_ngrams=1, 
		min_n=3, max_n=6, bucket=2000000,
		corpus_worker=3, corpus_chunksize=256, case_sensitive=False,
		window=5, min_count=5, max_vocab_size=None,
		sample=0.001, seed=1, workers=3, min_alpha=0.0001,
		sg=0, hs=0, compute_loss=False,negative=5, 
		ns_exponent=0.75, trim_rule=None, cbow_mean=1, 
		iter=5, null_word=0,  sorted_vocab=1, batch_words=10000):


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
			keywords, sentences, corpus_file,
			size, alpha, word_ngrams, min_n, max_n, bucket,
			corpus_worker, corpus_chunksize, case_sensitive,
			window, min_count, max_vocab_size, 
			sample, seed, workers, 
			min_alpha, sg, hs, compute_loss,
			negative, ns_exponent, cbow_mean, 
			iter, null_word, trim_rule, 
			sorted_vocab, batch_words)

		self.build_vocab(
			(corpus for corpus in KeywordCorpusIterator(self.kc)))



class KeywordCorpusFactoryGloveMixin(KeywordCorpusFactory):

	def __init__(
		# self, keywords, sentences, corpus_file, 
		# corpus_worker, corpus_chunksize, case_sensitive,
		# vocab_file, save_file, 
		# size, window, min_count, threads,
		# iters, X_max, memory,
		# update, pretrained_model_file, new_model_name

		# 20181128 Hannah Chen, modify class variables
		self, keywords, sentences, corpus_file, 
		corpus_worker, corpus_chunksize, case_sensitive,
		vocab_file, save_file, size, window, min_count, 
		threads, iter, X_max, memory, pretrained_model_file,
		output_file
		):

		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)
		self.kc = self.create(sentences, corpus_chunksize)
		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize    
		



#11/24 add 
# class SecGloVe(KeywordCorpusFactoryGloveMixin):

# 20181128 Hannah Chen, add inheritance from Sec2Vec
class SecGloVe(Sec2Vec, KeywordCorpusFactoryGloveMixin):

	def __init__(
		# self, keywords, sentences=None,
		# corpus_file=None, 
		# corpus_worker=3, corpus_chunksize=256, case_sensitive=False, 
		# vocab_file='vocab.txt', save_file='vector',
		# min_count=5, size=100, window=5, threads=3,
		# iters=5, X_max=10, memory=4.0, update=False,
		# pretrained_model_file=None, new_model_name=None

		# 20181128 Hannah Chen, modify class variables
		self, keywords, sentences=None,
		corpus_file=None, corpus_worker=3, corpus_chunksize=256, 
		case_sensitive=False, vocab_file='vocab.txt', save_file='vectors',
		min_count=5, size=100, window=5, threads=3, iter=5, 
		X_max=10, memory=4.0, pretrained_model_file=None, 
		output_file='./glove/glove_vectors_gensim.vec'
		):

		KeywordCorpusFactoryGloveMixin.__init__(
			# keywords, sentences,
			# corpus_file, corpus_worker,
			# corpus_chunksize, case_sensitive,
			# vocab_file, save_file,
			# min_count, size, window, threads,
			# iters, X_max, memory, update,
			# pretrained_model_file, new_model_name

			# 20181128 Hannah Chen, modify class variables
			self, keywords, sentences, corpus_file, 
			corpus_worker, corpus_chunksize, case_sensitive,
			vocab_file, save_file, size, window, min_count, threads, 
			iter, X_max, memory, pretrained_model_file, output_file)

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
		self.vector_size = size
		self.window = window
		self.threads = threads
		self.iter = iter
		self.X_max = X_max
		self.memory = memory
		# self.update = update
		self.pretrained_model_file = pretrained_model_file
		# self.new_model_name = new_model_name
		self.pre_trained_vec = None
		self.output_file = output_file
		self.model = None

		assert self.sentences or self.corpus_file

		if self.sentences and not self.corpus_file:
			sentences = (corpus for corpus in KeywordCorpusIterator(self.kc))
			f = open('./glove/temp_glove_sentence.txt', 'w+')

			for sentence in sentences:
				f.write(' '.join(sentence))
				f.write('\n')

			self.corpus_file = 'temp_glove_sentence.txt'

			f.close()


	# 20181128 Hannah Chen, add method for loading word vectors generated by GloVe
	def _load_glove_vec(self, glove_file):

		glove2word2vec(glove_input_file=glove_file, \
						word2vec_output_file=self.output_file)

		return KeyedVectors.load_word2vec_format(self.output_file, binary=False)


	# 20181128 Hannah Chen, modify train_embed method
	def train_embed(self, keywords=None, sentences=None, update=False, 
		epochs=None):

		epochs = epochs if epochs else self.iter

		if update:

			if isinstance(sentences, Iterator):
				raise ValueError(
					'sentences accpets list of str or list of tokens only.')

			self.update(keywords, SentenceIterator(sentences))

			if self.model:
				self.model.build_vocab(SentenceIterator(sentences), update=True)
				self.model.train(sentences, total_examples=self.model.corpus_count, \
									epochs=epochs)

			else:

				if not self.pre_trained_vec:
					pre_trained_vec = _load_pretrained_model('./glove/{}.txt'.format(self.save_file))
			
				new_model = Word2Vec(size=self.size, min_count=self.min_count)
				new_model.build_vocab(SentenceIterator(sentences))

				new_model.build_vocab([list(self.pre_trained_vec.vocab.keys())], update=update)
				new_model.intersect_word2vec_format(self.pretrained_model_file, binary=False, lockf=1.0)
				new_model.train(SentenceIterator(sentences), total_examples=new_model.corpus_count, epochs=epochs)

				self.model = new_model
				self.wv = new_model.wv
				del new_model

			self._cal_kv()

		else:

			argument = [
				'./demo_v2.sh', '--Corpus_File={}'.format(self.corpus_file),
				'--Save_File={}'.format(self.save_file), '--Vocab_File={}'.format(self.vocab_file),
				'--Vocab_Min_Count={}'.format(self.min_count), '--Vector_Size={}'.format(self.size),
				'--Window={}'.format(self.window), '--Threads={}'.format(self.threads),
				'--iters={}'.format(self.iter), '--X_max={}'.format(self.X_max),
				'--Memory={}'.format(self.memory)
			]
			process = subprocess.Popen(argument, stdin=PIPE, stdout=PIPE, cwd='glove/')
			
			for line in process.stdout:
				logging.info(line.decode('utf-8').strip())

			self.pre_trained_vec = self._load_glove_vec('./glove/{}.txt'.format(self.save_file))
			self.wv = self.pre_trained_vec.wv

			self._cal_kv()

	# def train_glove_embed(self):

	# 	if self.update and self.pretrained_model_file:  #是否需要update

	# 		if self.keywords:


	# 		if self.model:
	# 			self.model.build_vocab(SentenceIterator(sentences), update=True)
	# 			self.model.train(sentences, total_examples=self.model.corpus_count, \
	# 								epochs=self.model.epochs)

	# 		if isinstance(self.sentences, Iterator):
	# 			raise ValueError(
	# 				'sentences accpets list of str or list of tokens only.')

	# 		pre_trained_vec = _load_pretrained_model(self.pretrained_model_file)
			
	# 		new_model = Word2Vec(size=self.size, min_count=self.min_count)
	# 		new_model.build_vocab(sentences)
	# 		total_examples = new_model.corpus_count

	# 		new_model.build_vocab([list(pre_trained_vec.vocab.keys())], update=self.update)
	# 		new_model.intersect_word2vec_format(self.pretrained_model_file, binary=False, lockf=1.0)
	# 		new_model.train(sentences, total_examples=total_examples, epochs=self.iters)

	# 		self._cal_kv(new_model)

	# 		# 20181128 Hannah Chen
	# 		new_model.wv.save_word2vec_format(self.new_model_name + '.bin')
	# 		# new_model.save(self.new_model_name + '.bin')

	# 	else:

	# 		argument = [
	# 			'./demo_v2.sh', '--Corpus_File={}'.format(self.corpus_file),
	# 			'--Save_File={}'.format(self.save_file), '--Vocab_File={}'.format(self.vocab_file),
	# 			'--Vocab_Min_Count={}'.format(self.min_count), '--Vector_Size={}'.format(self.size),
	# 			'--Window={}'.format(self.window), '--Threads={}'.format(self.threads),
	# 			'--iters={}'.format(self.iters), '--X_max={}'.format(self.X_max),
	# 			'--Memory={}'.format(self.memory)
	# 		]
	# 		process = subprocess.Popen(argument, stdin=PIPE, stdout=PIPE, cwd='glove/')
			
	# 		for line in process.stdout:
	# 			logging.info(line.decode('utf-8').strip())

	# 		pre_trained_vec = _load_pretrained_model('./glove/vector.txt')

	# 		self._cal_kv(pre_trained_vec)

	def remove_temp_file(self):
		if self.corpus_file == 'temp_glove_sentence.txt':
			subprocess.run(['rm','-rf',self.corpus_file],cwd='glove/')

	# 20181128 Hannah Chen, add output SecGloVe method
	def save(self, new_model_name, binary=False):
		self.model.wv.save_word2vec_format(new_model_name + '.bin', binary=binary)

		



