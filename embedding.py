from gensim.models import Word2Vec

class SecWord2Vec(Word2Vec):

	def __init__(
		self, sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, 
		min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, 
		sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, iter=5, null_word=0, 
		trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False, max_final_vocab=None):

		super().__init__( 
			sentences=sentences, corpus_file=corpus_file, size=size, 
			alpha=alpha, window=window, min_count=min_count,
			max_vocab_size=max_vocab_size, sample=sample, seed=seed, 
			workers=workers, min_alpha=min_alpha, sg=sg, 
			hs=hs, negative=negative, ns_exponent=ns_exponent, 
			cbow_mean=cbow_mean, iter=iter, null_word=null_word, 
			trim_rule=trim_rule, sorted_vocab=sorted_vocab, batch_words=batch_words, 
			compute_loss=compute_loss, max_final_vocab=max_final_vocab)
		

	def train(
		self, sentences=None, corpus_file=None, total_examples=None, 
		total_words=None, epochs=None, start_alpha=None, end_alpha=None, 
		word_count=0, queue_factor=2, report_delay=1.0, compute_loss=False):

		self.train(
			sentences=sentences, corpus_file=corpus_file, 
			total_examples=total_examples, total_words=total_words, 
			epochs=epochs, start_alpha=start_alpha, 
			end_alpha=end_alpha, word_count=word_count, 
			queue_factor=queue_factor, report_delay=report_delay, 
			compute_loss=compute_loss)

	