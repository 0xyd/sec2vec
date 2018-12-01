from collections import Iterator

def assert_sentences(f):
	'''
	Normalize the input sentences' data structure
	'''

	def inner(self, k=None, sentences=None, corpus_file=None, *args): 
	# def inner(self, k=None, sentences=None, *args): 
		'''
		:param k: keywords or keyword
		:type k: str or list of str
		:param sentences: input sentences for training.
		:type sentences: str, list of str or list of listed tokens
		:param corpus_file: path for corpus
		:type corpus_file: str
		'''

		if isinstance(sentences, str):
			return f(self, k, [sentences], corpus_file, *args)

		elif isinstance(sentences, Iterator):
			return f(self, k, sentences, corpus_file, *args)

		elif isinstance(sentences, list):

			_sentences = []

			for s in sentences:

				if isinstance(s, str):
					_sentences.append(s)
				else:
					_sentences.append(' '.join(s))

			return f(self, k, _sentences, corpus_file, *args)

		return f(self, k, sentences, corpus_file, *args)

	return inner