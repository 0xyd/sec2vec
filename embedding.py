import os
import logging
import dill as pickle
import shlex
import subprocess
from itertools  import cycle
from subprocess import Popen
from subprocess import PIPE
from collections import Iterator
from multiprocessing import cpu_count

# import tqdm
import numpy as np
#from glove import Glove, Corpus
import dill as pickle
from gensim.models import KeyedVectors
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec

from logger import EpochLogger
from preprocessing import SentenceIterator
from preprocessing import KeywordCorpusFactory
from preprocessing import KeywordCorpusIterator


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Sec2Vec():


	def __init__(self, sentences, corpus_file):

		# 20181130 LIN, Y.D.: Error Message be shared across embeddings.
		if sentences is None and corpus_file is None:
			raise ValueError(
				'One of parameters, sentences and corpus_file should not be None.')

		# 20181130 LIN, Y.D.: Save all sentences for training
		if isinstance(sentences, Iterator):

			self.sentences = []
			for s in sentences: 
				self.sentences.append(s)

		else:
			self.sentences = sentences

		if sentences is None:
			self.corpus_file = corpus_file

	def __getitem__(self, word):

		try:
			return self.wv[word]
		except:
			return ValueError('{} does not exist.'.format(word))

	def _get_vec(self, token):

		if token in self.wv:
			return self.wv[token]
		else:
			return self.wv['<unk>']

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

	# 20181130 LIN, Y.D. Move from KeywordCorpusFactory
	def add_keyword_corpus(self, keyword, sentences):

<<<<<<< HEAD
=======
		#20181130 
		print(len(self.kc))
>>>>>>> upstream/master
		if isinstance(sentences, list):

			if keyword in self.kc:

				for s in sentences:

						self.kc[keyword].add(s)
						self.sentences.extend(sentences)
						self.sentences = list(set(self.sentences))


			else:
				print(self.sentences)
				self.kc[keyword] = set(sentences)
				self.sentences.extend(sentences)
				self.sentences = list(set(self.sentences))
		else:
			raise ValueError(
					'sentences accepts list only.')



	def train_embed(
		self, keywords=None, sentences=None, corpus_file=None, update=False,
		total_examples=None, total_words=None,  epochs=None, 
		start_alpha=None, end_alpha=None, word_count=0, 
		queue_factor=2, report_delay=1.0):

		epochs = epochs if epochs else self.epochs
		total_examples = total_examples if total_examples else self.corpus_count

		# FastText does not contain this variable
		compute_loss = self.compute_loss if hasattr(self, 'compute_loss') else False
	   
		if update:

			if isinstance(sentences, Iterator):
				raise ValueError(
					'sentences accpets list of str or list of tokens only.')

			self.build_vocab(SentenceIterator(sentences), update=update)
			self.update(keywords, SentenceIterator(sentences))

			# FastText does not contain this variable
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

			
			if compute_loss:
				self.train(
					# 20181130 LIN, Y.D. Train with all corpus
					SentenceIterator(self.sentences),
					# KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count, compute_loss,
					queue_factor, report_delay)

			else:

				self.train(
					# 20181130 LIN, Y.D. Train with all corpus
					SentenceIterator(self.sentences),
					# KeywordCorpusIterator(self.kc), 
					corpus_file, total_examples, total_words, epochs, 
					start_alpha, end_alpha, word_count,
					queue_factor, report_delay)

			self.wv['<unk>'] = np.random.uniform(-1, 1, (self.vector_size,))

			# 20181127 Hannah Chen, append word vector of 'unk' 
			# to the array that collects all word vectors
			if not compute_loss:
				self.wv.vectors_vocab = np.vstack((self.wv.vectors_vocab, self.wv['<unk>']))

		self._cal_kv()

	def save_embed(self, output_file_name):
		pickle.dump(self, open(output_file_name, 'wb'))

	def load_embed(input_file_name):
		return pickle.load(open(input_file_name, 'rb'))


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
		
		Sec2Vec.__init__(self, sentences, corpus_file)
		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)

		# 20181130 Hannah Chen
		self.kc = self.create(SentenceIterator(self.sentences), corpus_chunksize)
		# 20181130 LIN, Y.D.: Save all sentences for training
		# self.kc = self.create(self.sentences, corpus_chunksize)
		# self.kc = self.create(sentences, corpus_chunksize)

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
		self, keywords, sentences, corpus_file,
		size, alpha, word_ngrams, min_n, max_n, bucket,
		corpus_worker, corpus_chunksize, case_sensitive,
		window=5, min_count=5, max_vocab_size=None, 
		sample=0.001, seed=1, workers=3, 
		min_alpha=0.0001, sg=0, hs=0, 
		negative=5, ns_exponent=0.75, cbow_mean=1, 
		iter=5, null_word=0, trim_rule=None, 
		sorted_vocab=1, batch_words=10000):

		# 20181130 LIN, Y.D.: Save all sentences for training
		Sec2Vec.__init__(self, sentences, corpus_file)

		# 20181126 Hannah Chen, modified variable: corpus_worker
		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)

		# 20181130 Hannah Chen
		self.kc = self.create(SentenceIterator(self.sentences), corpus_chunksize)
		# 20181130 LIN, Y.D.: Save all sentences for training
		# self.kc = self.create(self.sentences, corpus_chunksize)
		# self.kc = self.create(sentences, corpus_chunksize)

		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))

		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize

		FastText.__init__(self, 
			corpus_file=corpus_file, size=size, alpha=alpha, word_ngrams=word_ngrams,
			window=window, min_count=min_count, max_vocab_size=max_vocab_size, 
			sample=sample, seed=seed, workers=workers, min_n=min_n, max_n=max_n,
			min_alpha=min_alpha, sg=sg, hs=hs, bucket=bucket,
			negative=negative, ns_exponent=ns_exponent, cbow_mean=cbow_mean, 
			iter=iter, null_word=null_word, trim_rule=trim_rule, 
			sorted_vocab=sorted_vocab, batch_words=batch_words,
			callbacks=[]) 


class SecWord2Vec(KeywordCorpusFactoryWord2VecMixin):

	def __init__(
		self, keywords, sentences, 
		corpus_worker=3, corpus_chunksize=5000, case_sensitive=False, 
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
			(corpus for corpus in SentenceIterator(self.sentences)))

class SecFastText(KeywordCorpusFactoryFasttextMixin):

	def __init__(
		self, keywords, sentences, corpus_file=None,
		size=100, alpha=0.025, word_ngrams=1, 
		min_n=3, max_n=6, bucket=2000000,
		corpus_worker=3, corpus_chunksize=5000, case_sensitive=False,
		window=5, min_count=5, max_vocab_size=None,
		sample=0.001, seed=1, workers=3, min_alpha=0.0001,
		sg=0, hs=0, negative=5, 
		ns_exponent=0.75, trim_rule=None, cbow_mean=1, 
		iter=5, null_word=0,  sorted_vocab=1, batch_words=10000):


		super().__init__(
			keywords, sentences, corpus_file,
			size, alpha, word_ngrams, min_n, max_n, bucket,
			corpus_worker, corpus_chunksize, case_sensitive,
			window, min_count, max_vocab_size, 
			sample, seed, workers, 
			min_alpha, sg, hs, 
			negative, ns_exponent, cbow_mean, 
			iter, null_word, trim_rule, 
			sorted_vocab, batch_words)

		self.build_vocab(
			(corpus for corpus in SentenceIterator(self.sentences)))


class KeywordCorpusFactoryGloveMixin(Sec2Vec, KeywordCorpusFactory):

	def __init__(
		self, keywords, sentences, corpus_file, 
		corpus_worker, corpus_chunksize, case_sensitive
		):

		Sec2Vec.__init__(self, sentences, corpus_file)
		KeywordCorpusFactory.__init__(self, keywords, case_sensitive, corpus_worker)

		# 20181130 Hannah Chen
		self.kc = self.create(SentenceIterator(self.sentences), corpus_chunksize)
		self.kv = dict(((keyword, []) for keyword in self.kc.keys()))
		self.keyword_count = dict(((keyword, 0) for keyword in self.kc.keys()))
		self.corpus_chunksize = corpus_chunksize
		

#11/24 add 
class SecGloVe(KeywordCorpusFactoryGloveMixin):

	def __init__(
		self, keywords, sentences=None,
		corpus_file=None, corpus_worker=3, corpus_chunksize=5000, 
		case_sensitive=False, vocab_file='vocab.txt', save_file='vectors',
		min_count=5, size=100, window=5, threads=3, iter=5, 
		X_max=10, memory=4.0, pretrained_model_file=None, 
		output_file='glove_vectors_gensim.vec',
		verbose=2, binary=2, cooccurrence_file='cooccurrence.bin',
		cooccurrence_shuf_file='cooccurrence.shuf.bin' ,
		builddir='build', glove_dir='Glove/'
		):

		# 20181130 LIN, Y.D.
		super().__init__(
			keywords, sentences, corpus_file, 
			corpus_worker, corpus_chunksize, case_sensitive)

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
		self.pre_trained_vec = None

		# 20181129 Hannah Chen
		self.glove_dir = glove_dir
		self.output_file = '{}/{}'.format(glove_dir, output_file)
		self.model = None

		#20181229 arvis add variables
		self.verbose = verbose
		self.binary = binary
		self.cooccurrence_file = cooccurrence_file
		self.cooccurrence_shuf_file = cooccurrence_shuf_file
		self.builddir = builddir
		self.glove_dir = glove_dir

		if self.sentences:
			
			f = open('./{}/temp_glove_sentence.txt'.format(self.glove_dir), 'w+')

			for sentence in SentenceIterator(self.sentences):
				
				f.write(' '.join(sentence))
				f.write('\n')

			self.corpus_file = 'temp_glove_sentence.txt'

			f.close()

		elif self.corpus_file:

			os.system('cp ./{} ./{}/'.format(self.corpus_file, self.glove_dir))
			f = open('./{}'.format(self.corpus_file), 'r')
			self.sentences = f.readlines()

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

				glove_vec_file = '{}/{}.txt'.format(self.glove_dir, self.save_file)

				if not self.pre_trained_vec:
					self.pre_trained_vec = self._load_pretrained_model('{}/{}.txt'.format(self.glove_dir ,self.save_file))
			
				new_model = Word2Vec(
					SentenceIterator(sentences), 
					size=self.size, 
					min_count=self.min_count)

				new_model.build_vocab(
					[list(self.pre_trained_vec.vocab.keys())], 
					update=update)
				new_model.intersect_word2vec_format(
					self.output_file, binary=False, lockf=1.0)
				new_model.train(
					SentenceIterator(sentences), 
					total_examples=new_model.corpus_count, 
					epochs=epochs)

				self.model = new_model
				self.wv = new_model.wv
				del new_model

			self._cal_kv()

		else:

			# 20181130 LIN, Y.D. Update Naming
			vocab_count_cmd = '{}/vocab_count -min-count {} -verbose {} '\
								.format(self.builddir, self.min_count, self.verbose)

			cooccur_cmd = '{}/cooccur -memory {} -vocab-file {} -verbose {} -window-size {}'.format(
				self.builddir, self.memory, self.vocab_file, self.verbose, self.window)

			shuffle_cmd = '{}/shuffle -memory {} -verbose {} '.format(
				self.builddir, self.memory, self.verbose)

			save_file_cmd = '''{}/glove -save-file {} -threads {} -input-file {}
							-x-max {} -iter {} -vector-size {} -binary {} -vocab-file {} -verbose {}'''.format(
								self.builddir, self.save_file, self.threads, self.cooccurrence_shuf_file,\
								self.X_max, self.iter, self.size, self.binary, self.vocab_file, self.verbose)

			glove_command = [
				(vocab_count_cmd, self.corpus_file, self.vocab_file, True, True),
				(cooccur_cmd, self.corpus_file, self.cooccurrence_file, True, True),
				(shuffle_cmd, self.cooccurrence_file, self.cooccurrence_shuf_file, True, True),
				(save_file_cmd, None, None, False, False) 
			]

			for command in glove_command:
				self._run_subprocess_command(*command)

			self._remove_temp_file()

			self.pre_trained_vec = self._load_glove_vec('{}/{}.txt'.format(self.glove_dir, self.save_file))
			self.wv = self.pre_trained_vec.wv

			self._cal_kv()
			logging.info('end to embedding...')


	def _run_subprocess_command(
		self, command, 
		input_path=None, output_path=None, 
		input_enable=False, output_enable=False):

		if input_enable == output_enable == True:
			input_file = open(self.glove_dir + input_path)
			output_file = open(self.glove_dir + output_path, 'wb')

			with Popen(
				shlex.split(command), stdin=input_file, 
				stdout=PIPE, cwd=self.glove_dir) as p:
				for l in p.stdout:
					output_file.write(l)
					output_file.flush()
				output_file.close()

		else:
			with Popen(shlex.split(command), stdin=PIPE, stdout=PIPE, cwd=self.glove_dir) as p:
				for line in p.stdout:
					logging.info(line) 


	def _remove_temp_file(self):

		if self.corpus_file:
			os.remove('{}/{}'.format(self.glove_dir ,self.corpus_file))
