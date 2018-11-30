import gc
import sys
import time
import math
from multiprocessing import Pool, cpu_count
from queue import Queue

import torch
import torch.nn as nn
import numpy  as np
# import pandas as pd
from tqdm import tqdm


def get_vec(word, *arr): return word, np.asarray(arr, dtype='float32')


def cal_word_index(embeddings):

	word_set = set()

	for embedding in embeddings:

		for word in embedding.keys():
			word_set.add(word)

	word_index = dict((word, index) for index, word in enumerate(word_set))

	return word_index


class ConvNet(nn.Module):
	
	def __init__(self, num_words, num_embeddings, channels,
				 input_embedding_size, output_embedding_size):
		
		super().__init__()
		
		self.conv_1 = nn.Sequential(
			nn.Conv1d(num_embeddings, channels[0], kernel_size=3),
			nn.MaxPool1d(3))
		
		seq_len = input_embedding_size
		for nn_layer in self.conv_1:
			seq_len = self._cal_seq_length(
				nn_layer, seq_len)
		
		self.conv_2 = nn.Sequential(
			nn.Conv1d(channels[0], channels[1], kernel_size=3),
			nn.MaxPool1d(3))
		
		for nn_layer in self.conv_2:
			seq_len = self._cal_seq_length(nn_layer, seq_len)
			
		self.fc = nn.Sequential(
			nn.Linear(seq_len*channels[-1], output_embedding_size),
			nn.Linear(output_embedding_size, num_words))
		
		self.out = nn.Softmax(1)
		
	def _cal_seq_length(self, nn_layer, seq_len):
		
		if isinstance(nn_layer, nn.Conv1d):
			
			in_channels = nn_layer.in_channels
			
			padding = 0
			for p in nn_layer.padding:
				padding += p
			
			kernel_size = 0
			for k in nn_layer.kernel_size:
				kernel_size += k
				
			stride = 0
			for s in nn_layer.stride:
				stride += s
			
			dilation = 0
			for d in nn_layer.dilation:
				dilation += d
				
			return self._cal_conv1_seq_length(
				seq_len, kernel_size, stride, padding, dilation)
		
		elif isinstance(nn_layer, nn.MaxPool1d):
			
			return self._cal_max_pool_seq_length(
				seq_len, nn_layer.kernel_size, 
				nn_layer.stride, nn_layer.padding, nn_layer.dilation)
		
	
	def _cal_conv1_seq_length(self, seq_len, kernel_size, stride, padding, dilation):
		return math.floor((
			(seq_len + 2*padding - dilation * (kernel_size-1) - 1) / stride) + 1)
	
	def _cal_max_pool_seq_length(self, seq_len, kernel_size, stride, padding, dilation):
		return math.floor((
			(seq_len+ 2*padding - dilation*(kernel_size-1)) / stride) + 1)
	
	def forward(self, x):
		
		out = self.conv_1(x)
		out = self.conv_2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		out = self.out(out)
		
		return out
		


class CNNInfusion():
	
	def __init__(self, epochs, batch_size, embedding_size):
		'''
		'''
		
		self.iv = dict()
		self.epochs = epochs
		self.batch_size = batch_size
		self.embedding_size = embedding_size
		self.criterion = nn.BCELoss()
		self.device = torch.device(
			'cuda:0' if torch.cuda.is_available() else 'cpu')
		
	def _build_word_index(self, shared_embeddings):
		'''
		'''
		
		words = set()
		
		for data in shared_embeddings.values():
			for sentence in data['corpus']:
				for token in sentence.split(' '):
					words.add(token)

		return words
	
	
	def _get_shared_keyword_corpus(self, embeddings):
		'''
		Get shared keywords and corpus across different embeddings.
		'''
		
		shared_keywords = set()
		
		for i, e in enumerate(embeddings):
			
			if i:
				shared_keywords = shared_keywords.union(set(e.kv.keys()))
			else:
				shared_keywords = set(e.kv.keys())
			
		shared_embedding = dict()
		
		for keyword in shared_keywords:
			
			corpus = None
			
			# Corpus must be the same otherwise infusion is invalid.
			for i, e in enumerate(embeddings):
				
				if i:
					if corpus != e.kc[keyword]:
						print(
							'Different embeddinga should share the same corpus for keyword {}'.format(keyword))
						break
				else:
					corpus = e.kc[keyword]
					
			shared_embedding[keyword] = dict()
			shared_embedding[keyword]['corpus'] = corpus
			shared_embedding[keyword]['vector'] = []
			
			# Collect corpus which is embedding by different models
			for e in embeddings:
				shared_embedding[keyword]['vector'].append(e.kv[keyword])
		
		return shared_embedding
		
	
	def _cnn_train(self, cnn, optimizer, word_index, embeddings):
		'''
		'''
		
		num_output_words = len(word_index)
		num_keywords = len(embeddings)
		
		for epoch in tqdm(range(self.epochs)):
			
			batch = []
			for keyword_index, (keyword, data) in tqdm(
				enumerate(embeddings.items()), total=num_keywords):
				
				tokens = set(
					[s for sentence in data['corpus'] for s in sentence.split(' ')])
				tokens_index = [
					word_index[token] if token in word_index else word_index['<unk>'] 
						for token in tokens]
				batch.append((tokens, tokens_index, data['vector']))
				
				if len(batch) == self.batch_size:

					feature_map = torch.tensor(
						np.array([b[2] for b in batch])).to(
							self.device, dtype=torch.float32)
					lbls = torch.tensor(
						np.zeros((self.batch_size, num_output_words))).to(
							self.device, dtype=torch.float32)
			
					for i, index in enumerate((b[1] for b in batch)):
						lbls[i, index] = 1.
					
					outputs = cnn(feature_map)
					loss = self.criterion(outputs, lbls)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					
					if (keyword_index+1) % (self.batch_size*100) == 0:
						print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
							  .format(epoch+1, self.epochs, keyword_index, num_keywords, loss.item()))
						
					batch = []
		
	def train(self, word_batch, embeddings, channels, seed=42,
			  input_embedding_size=100, output_embedding_size=100, learning_rate=0.001):
		'''
		'''
		
		num_embeddings = len(embeddings)
		data = self._get_shared_keyword_corpus(embeddings)
		total_words = self._build_word_index(data)
		total_words_num = len(total_words)
		full_batch_num  = total_words_num // word_batch
		last_batch_size = total_words_num % word_batch
		
		# Initialize random seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)

		word_index = dict()
		
		for w_idx, w in tqdm(
			enumerate(total_words), total=total_words_num):
			
			word_index[w] = w_idx % word_batch
			word_index_size = len(word_index)
			run_batch = False
			
			if word_index_size == word_batch: 
				run_batch = True
			elif w_idx >= full_batch_num*word_batch and word_index_size == last_batch_size:
				run_batch = True
				
			if run_batch:
			
				if '<unk>' not in word_index: word_index['<unk>'] = len(word_index)
				num_words = len(word_index)

				cnn = ConvNet(num_words, num_embeddings, channels,
							  input_embedding_size, output_embedding_size).to(self.device)
			
				optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

				self._cnn_train(cnn, optimizer, word_index, data)
				
				for word, index in word_index.items():
					self.iv[word] = cnn.fc[-1].weight.data[index]
				
				del cnn; gc.collect()
				del word_index; gc.collect()


				if torch.cuda.is_available():
					torch._C._cuda_emptyCache()
				
				word_index = dict()
				
	
	


# def cal_max_string(word_index):

# 	max_word_size = 0

# 	for word in words_index.keys():
	
# 		word_size = len(word.encode('utf8'))
	
# 		if word_size > max_word_size:
# 			max_word_size = word_size
		
# 	return max_word_size


# Reconsidering ...
# def mp_dump_vec(
# 	store, embedding_name, words, vecs, 
# 	feature_cols, max_word_size, append=True):

# 	df = pd.DataFrame(dict(word=words,value=vecs))
# 	df[feature_cols] = pd.DataFrame(
# 		df['value'].values.tolist(), 
# 		index=df.index)
# 	df.drop(['value'], inplace=True, axis=1)

# 	store.append(
# 		embedding_name, df, 
# 		append=append, min_itemsize={'word': max_word_size})
	
# 	del df; gc.collect()


# def dump_embeddings(
# 	word_index, max_word_size, total_word_num, 
# 	embeddings, embedding_size, embedding_store_path, 
# 	word_batch=1000, workers=cpu_count()):

# 	embedding_names = [e['name'] for e in embeddings]
# 	embedding_dict  = dict(((e['name'], []) for e in embeddings))
# 	embedding_dict['word'] = []
# 	feature_cols = ['feature_{}'.format(i) for i in range(embedding_size)]
# 	store = pd.HDFStore(embedding_store_path)
# 	# worker_pool = Pool(workers)
# 	work_queue = Queue()

# 	for i, word in tqdm(enumerate(word_index.keys()), total=total_word_num):

# 		for embedding in embeddings:

# 			_embedding = embedding['obj']

# 			if word in _embedding:
# 				_embed_val = _embedding[word]
# 			elif 'unk' in embedding:
# 				_embed_val = _embedding['unk']
# 			else:
# 				_embed_val = np.random.uniform(-1, 1, (embedding_size,))

# 			embedding_dict[embedding['name']].append(_embed_val)

# 		embedding_dict['word'].append(word)

# 		if (i+1) % word_batch == 0 and i > 0:

# 			if i+1 == word_batch:

# 				for name in embedding_names:
# 					mp_dump_vec(store, name, 
# 						embedding_dict['word'], embedding_dict['name'],
# 						feature_cols, max_word_size, False)
# 			else:

# 				for name in embedding_names:
# 					work_queue.put(
# 						store, name, 
# 						embedding_dict['word'], embedding_dict['name'],
# 						feature_cols, max_word_size, False)

# 				while not work_queue.empty():

# 					with Pool(workers) as worker_pool:
# 						work = work_queue.get()
# 						pool.apply_async(
# 							mp_dump_vec, 
# 							(
# 								store, name, 
# 								embedding_dict['word'], embedding_dict['name'],
# 								feature_cols, max_word_size, False))

# 			del embedding_dict; gc.collect()

# 			embedding_dict = dict(((e['name'], []) for e in embeddings))
# 			embedding_dict['word'] = []




