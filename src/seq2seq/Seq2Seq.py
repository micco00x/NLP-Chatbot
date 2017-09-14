import torch
import numpy as np
import random

from seq2seq import Encoder, Decoder, AttnDecoderRNN

# Inspired by http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and https://github.com/tensorflow/nmt
class Seq2Seq:
	def __init__(self,
				 #encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
				 input_vocabulary_dim, target_vocabulary_dim, #target_max_length,
				 go_symbol_idx, eos_symbol_idx,
				 embedding_dim,
				 embedding_matrix_encoder=None, embedding_matrix_decoder=None,
				 embedding_padding_idx=None):
		# hparams:
		#encoder_input_size = input_vocabulary_dim
		encoder_hidden_size = 4096
		encoder_n_layers = 1
		
		decoder_hidden_size = 4096
		#decoder_output_size = target_vocabulary_dim
		decoder_n_layers = 1
		
		#self.target_max_length = target_max_length
		
		self.GO_SYMBOL_IDX = go_symbol_idx
		self.EOS_SYMBOL_IDX = eos_symbol_idx
	
		# Encoder:
		self.encoder = Encoder.Encoder(input_vocabulary_dim,
									   embedding_dim,
									   encoder_hidden_size,
									   encoder_n_layers,
									   embedding_matrix_encoder,
									   embedding_padding_idx)
									   
		# Decoder:
		self.decoder = Decoder.Decoder(target_vocabulary_dim,
									   embedding_dim,
									   decoder_hidden_size,
									   decoder_n_layers,
									   embedding_matrix_decoder,
									   embedding_padding_idx)
		#self.decoder = AttnDecoderRNN.AttnDecoderRNN(decoder_hidden_size,
		#											 decoder_output_size,
		#											 self.target_max_length,
		#											 decoder_n_layers,
		#											 0.1,
		#											 embedding_matrix_decoder)
									   
		if torch.cuda.is_available():
			self.encoder = self.encoder.cuda()
			self.decoder = self.decoder.cuda()
											   
		# Optimizers:
		self.encoder_optimizer = torch.optim.RMSprop(self.encoder.parameters())
		self.decoder_optimizer = torch.optim.RMSprop(self.decoder.parameters())
		
		# Loss (embedding_padding_idx is ignored, it does not contribute to input gradients):
		self.criterion = torch.nn.NLLLoss(ignore_index=embedding_padding_idx)

	def train(self, X, Y, batch_size): # TODO: add X_dev=None, Y_dev=None, batch_size
		
		# TODO: add padding when using batches
		cnt = 0
		#for x, y in zip(X, Y):
		for idx in range(0, len(X), batch_size):
			x = np.array(X[idx:min(idx+batch_size, len(X))])
			y = np.array(Y[idx:min(idx+batch_size, len(Y))])
			
			x = torch.autograd.Variable(torch.LongTensor(x))
			y = torch.autograd.Variable(torch.LongTensor(y))
			
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()
		
			#batch_size = x.size()[0]
			
			cnt += 1
			
			encoder_hidden = self.encoder.initHidden(x.size()[0])
			
			self.encoder_optimizer.zero_grad()
			self.decoder_optimizer.zero_grad()

			input_length = x.size()[1]
			target_length = y.size()[1]

			#encoder_outputs = torch.autograd.Variable(torch.zeros(self.target_max_length, self.encoder.hidden_size))
			#encoder_outputs = encoder_outputs.cuda() if torch.cuda.is_available() else encoder_outputs

			loss = 0

			encoder_output, encoder_hidden = self.encoder(x, encoder_hidden)

			decoder_input = torch.autograd.Variable(torch.LongTensor([[self.GO_SYMBOL_IDX] * x.size()[0]]))
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			
			decoder_hidden = encoder_hidden

			for di in range(target_length):
				decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
				topv, topi = decoder_output.data.topk(1)
			
				decoder_input = torch.autograd.Variable(torch.LongTensor(topi))
				decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
			
				loss += self.criterion(decoder_output, y[:,di])
			
				#print("topi:")
				#print(topi)
				#print("y[:,di]:")
				#print(y[:,di])

			tot_loss = loss.data[0] / x.size()[0]
			print("Avg. loss at iteration " + str(cnt) + ": " + str(tot_loss))

			loss.backward()

			self.encoder_optimizer.step()
			self.decoder_optimizer.step()

			#print("Loss (" + str(cnt) + "): " + str(loss), end="\r")
			
