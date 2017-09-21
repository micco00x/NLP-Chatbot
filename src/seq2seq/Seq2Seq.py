import torch
import utils

#########################################################################################
# Inspired by:
#     [1] http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#     [2] https://github.com/tensorflow/nmt
#########################################################################################

class Seq2Seq(torch.nn.Module):
	def __init__(self,
				 mode,
				 #encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
				 input_vocabulary_dim, target_vocabulary_dim, #target_max_length,
				 pad_symbol_idx, go_symbol_idx, eos_symbol_idx,
				 encoder_hidden_size, decoder_hidden_size,
				 embedding_dim,
				 embedding_matrix_encoder=None, embedding_matrix_decoder=None,
				 embedding_padding_idx=None,
				 n_layers=1, bidirectional=False):
		
		super(Seq2Seq, self).__init__()
		
		# TODO: with mode="eval" there's no need to take values of gradients
		#		of gradients of the graph since a backward operations is never
		#		performed, you can use this to save RAM
		if mode == "eval":
			self.volatile = True
		else:
			self.volatile = False
		
		# hparams:
		self.mode = mode
		self.target_vocabulary_dim = target_vocabulary_dim
		self.embedding_padding_idx = embedding_padding_idx
		#bidirectional = False
		#n_layers = 1
		
		#encoder_input_size = input_vocabulary_dim
		self.encoder_hidden_size = encoder_hidden_size
		self.decoder_hidden_size = decoder_hidden_size
		#decoder_output_size = target_vocabulary_dim
		
		#self.target_max_length = target_max_length
		
		self.PAD_SYMBOL_IDX = pad_symbol_idx
		self.GO_SYMBOL_IDX = go_symbol_idx
		self.EOS_SYMBOL_IDX = eos_symbol_idx
	
		# Encoder:
		self.encoder_embedding = torch.nn.Embedding(input_vocabulary_dim, embedding_dim, padding_idx=embedding_padding_idx)
		if embedding_matrix_encoder is not None:
			self.encoder_embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix_encoder).float())

		self.encoder_gru = torch.nn.GRU(embedding_dim, self.encoder_hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
									   
		# Decoder:
		self.decoder_embedding = torch.nn.Embedding(target_vocabulary_dim, embedding_dim, padding_idx=embedding_padding_idx)
		if embedding_matrix_decoder is not None:
			self.decoder_embedding.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix_decoder).float())
	
		self.decoder_gru = torch.nn.GRU(embedding_dim, self.decoder_hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
		self.decoder_out = torch.nn.Linear(self.decoder_hidden_size, target_vocabulary_dim)
		self.softmax = torch.nn.LogSoftmax()

	# encoder_input shape: (batch_size, encoder_seq_len)
	# decoder_input shape: (batch_size, 1)
	# output shape (list of tensors): (batch_size, target_length, target_vocabulary_dim)
	# if target_length=None then forward will be treated as in eval mode (i.e.
	# outpus shape will be of shape (1, length), which is the answer to the question)
	def forward(self, encoder_input, decoder_input, target_length):
		
		batch_size = encoder_input.size()[0]
		
		output_data = []

		encoder_output, encoder_hidden = self._encoder_forward(encoder_input)

		#decoder_input = torch.autograd.Variable(torch.LongTensor([[self.GO_SYMBOL_IDX] * encoder_input.size()[0]]))
		#decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

		decoder_hidden = encoder_hidden

		if self.mode == "train":
			for di in range(target_length):
				decoder_output, decoder_hidden = self._decoder_forward(decoder_input, decoder_hidden)
				topv, topi = decoder_output.data.topk(1)

				decoder_input = torch.autograd.Variable(torch.LongTensor(topi.cpu()), volatile=self.volatile)
				decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

				output_data.append(decoder_output)
		else:
			l = 0
			while decoder_input[0].data[0] != self.EOS_SYMBOL_IDX:
				decoder_output, decoder_hidden = self._decoder_forward(decoder_input, decoder_hidden)
				topv, topi = decoder_output.data.topk(1)

				decoder_input = torch.autograd.Variable(torch.LongTensor(topi.cpu()), volatile=self.volatile)
				decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
				
				#output_data.append(decoder_input[0].data[0])
				output_data.append(decoder_output)
				
				# Break if the network loops the answer:
				l += 1
				if l > target_length:
					break
	
		return output_data

	def _encoder_forward(self, input):
		embedded = self.encoder_embedding(input)
		return self.encoder_gru(embedded)

	def _decoder_forward(self, input, hidden):
		output = self.decoder_embedding(input)
		output = torch.nn.functional.relu(output)
		output, hidden = self.decoder_gru(output, hidden)
		output = self.decoder_out(output[:,0,:self.decoder_hidden_size])
		output = self.softmax(output)
		return output, hidden
