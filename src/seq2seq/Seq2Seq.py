import torch

from seq2seq import Encoder, Decoder

# Inspired by http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and https://github.com/tensorflow/nmt
class Seq2Seq:
	def __init__(self,
				 #encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
				 word2vec):
		# hparams:
		encoder_input_size = word2vec.VOCABULARY_DIM
		encoder_hidden_size = word2vec.EMBEDDING_DIM
		encoder_n_layers = 1
		
		decoder_hidden_size = word2vec.EMBEDDING_DIM
		decoder_output_size = word2vec.VOCABULARY_DIM
		decoder_n_layers = 1
		
		# TODO: move GO_SYMBOL, EOS_SYMBOL, embedding_matrix, etc. as hparams
	
		# Encoder:
		self.encoder = Encoder.Encoder(encoder_input_size,
									   encoder_hidden_size,
									   encoder_n_layers,
									   word2vec.embedding_matrix)
									   
		# Decoder:
		self.decoder = Decoder.Decoder(decoder_hidden_size,
									   decoder_output_size,
									   decoder_n_layers,
									   word2vec.embedding_matrix)
											   
		# Optimizers:
		self.encoder_optimizer = torch.optim.RMSprop(self.encoder.parameters())
		self.decoder_optimizer = torch.optim.RMSprop(self.decoder.parameters())
		
		# Loss:
		self.criterion = torch.nn.NLLLoss()
		
		self.word2vec = word2vec

	def train(self, X, Y, epochs): # TODO: add X_dev=None, Y_dev=None, batch_size
		for epoch in range(epochs):
			# TODO: add padding when using batches
			print("Epoch", epoch+1)
			cnt = 0
			for x, y in zip(X, Y):
				cnt += 1
			
				encoder_hidden = self.encoder.initHidden()
				
				self.encoder_optimizer.zero_grad()
				self.decoder_optimizer.zero_grad()

				input_length = x.size()[0]
				target_length = y.size()[0]

				#encoder_outputs = torch.autograd.Variable(torch.zeros(MAX_LENGTH, self.encoder.hidden_size)) # TODO

				loss = 0

				for ei in range(input_length):
					encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
					#encoder_outputs[ei] = encoder_output[0][0]

				decoder_input = torch.autograd.Variable(torch.LongTensor([[self.word2vec.GO_SYMBOL]]))
				decoder_hidden = encoder_hidden

				# Without teacher forcing:
				for di in range(target_length):
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
					topv, topi = decoder_output.data.topk(1)
					ni = topi[0][0]

					decoder_input = torch.autograd.Variable(torch.LongTensor([[ni]]))

					loss += self.criterion(decoder_output, y[di])
					if ni == self.word2vec.EOS_SYMBOL:
						break

				loss.backward()

				self.encoder_optimizer.step()
				self.decoder_optimizer.step()

				print("Loss (" + str(cnt) + "): " + str(loss), end="\r")
			print("Loss:", loss)
			
