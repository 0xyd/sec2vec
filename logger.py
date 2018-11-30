from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):

	def __init__(self, compute_loss):
		print("Epoch Logger is prepared.")
		self.epoch = 0
		self.compute_loss = compute_loss

	def on_epoch_begin(self, model):
		print("Epoch #{} start".format(self.epoch))

	def on_epoch_end(self, model):

		# 20181126 Hannah Chen, print training loss if compute_loss is True
		if hasattr(self, 'compute_loss'):
			print("Epoch #{} end - training loss: {}".format(self.epoch, model.get_latest_training_loss()))
		else:
			print("Epoch #{} end".format(self.epoch))	

		self.epoch += 1