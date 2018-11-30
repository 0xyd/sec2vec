from collections import Iterator

def assert_sentences(f):

	def inner(self, k=None, sentences=None, *args):

		if isinstance(sentences, str):
			return f(self, k, [sentences], *args)

		elif isinstance(sentences, Iterator):
			return f(self, k, sentences, *args)

		elif isinstance(sentences, list):

			_sentences = []

			for s in sentences:

				if isinstance(s, str):
				# if not isinstance(s, str):
				# 	raise ValueError('Sentences must be list of string or list of listed tokens.')
				# elif isinstance(s, list):
					_sentences.append(s)
				else:
					_sentences.append(' '.join(s))

			return f(self, k, _sentences, *args)

		return f(self, k, sentences, *args)

	return inner