import torch

from seq2seq import Encoder, Decoder

# Inspired by http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and https://github.com/tensorflow/nmt
class Seq2Seq:
	def __init__(self,
				 #encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
				 vocabulary_dim, go_symbol_idx, eos_symbol_idx,
				 embedding_dim, embedding_matrix_encoder, embedding_matrix_decoder):
		# hparams:
		encoder_input_size = vocabulary_dim
		encoder_hidden_size = embedding_dim
		encoder_n_layers = 1
		
		decoder_hidden_size = embedding_dim
		decoder_output_size = vocabulary_dim
		decoder_n_layers = 1
		
		self.GO_SYMBOL_IDX = go_symbol_idx
		self.EOS_SYMBOL_IDX = eos_symbol_idx
	
		# Encoder:
		self.encoder = Encoder.Encoder(encoder_input_size,
									   encoder_hidden_size,
									   encoder_n_layers,
									   embedding_matrix)
									   
		# Decoder:
		self.decoder = Decoder.Decoder(decoder_hidden_size,
									   decoder_output_size,
									   decoder_n_layers,
									   embedding_matrix)
											   
		# Optimizers:
		self.encoder_optimizer = torch.optim.RMSprop(self.encoder.parameters())
		self.decoder_optimizer = torch.optim.RMSprop(self.decoder.parameters())
		
		# Loss:
		self.criterion = torch.nn.NLLLoss()

	def train(self, X, Y, epochs): # TODO: add X_dev=None, Y_dev=None, batch_size
		print_every = 500
		for epoch in range(epochs):
			# TODO: add padding when using batches
			print("Epoch", epoch+1)
			cnt = 0
			tot_loss = 0
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

				decoder_input = torch.autograd.Variable(torch.LongTensor([[self.GO_SYMBOL_IDX]]))
				decoder_hidden = encoder_hidden

				# Without teacher forcing:
				for di in range(target_length):
					decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
					topv, topi = decoder_output.data.topk(1)
					ni = topi[0][0]

					decoder_input = torch.autograd.Variable(torch.LongTensor([[ni]]))

					loss += self.criterion(decoder_output, y[di])
					if ni == self.EOS_SYMBOL_IDX:
						break

				tot_loss += loss.data[0]
				if cnt % print_every == 0:
					print("Avg. loss at iteration " + str(cnt) + ": " + str(tot_loss/print_every))
					tot_loss = 0

				loss.backward()

				self.encoder_optimizer.step()
				self.decoder_optimizer.step()

				#print("Loss (" + str(cnt) + "): " + str(loss), end="\r")
			print("Loss:", loss)
			
