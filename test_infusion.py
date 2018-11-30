import pytest
import numpy as np

from infusion import CNNInfusion
from embedding import SecWord2Vec
from embedding import SecFastText
from embedding import SecGloVe

class TestCNNInfusion():

	def test_train(self):

		test_samples = [
			'This is a Hello World Sample',
			'this is a hello world sample',	
			'is this a hello world sample',
			'Is this a hello world sample',
			'Is this a Hello World Sample',
			'is This a hello World sample',
			'IS THIS A HELLO WORLD SAMPLE',
			'is this a hello world sample',
		]

		keywords = [
			'this', 'is', 'a', 'hello', 'world', 'sample', 
			'THIS', 'IS', 'A', 'HELLO', 'WORLD', 'SAMPLE',
		]

		w2v = SecWord2Vec(keywords, test_samples, min_count=1, case_sensitive=True)
		ft  = SecFastText(keywords, test_samples, min_count=1, case_sensitive=True)
		gv  = SecGloVe(keywords, test_samples, min_count=1, verbose=0, case_sensitive=True)

		w2v.train_embed()
		ft.train_embed()
		gv.train_embed()

		cnn_infusion = CNNInfusion(10, 2, 100)
		cnn_infusion.train(2, [w2v, ft, gv], [64, 4])

		# Infusion vector must have the same number of embedding words
		words_set = set()
		for sample in test_samples:
			for token in sample.split(' '):
				words_set.add(token)

		assert len(cnn_infusion.iv) == len(words_set)+1

		# Infusion vector must not be nan
		for n, v in cnn_infusion.iv.items():
			assert np.where(np.isnan(v))[0].shape[0] == 0

