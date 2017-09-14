import torch

from seq2seq import Encoder, Decoder, AttnDecoderRNN

# Inspired by http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# and https://github.com/tensorflow/nmt

class Seq2Seq:
	def __init__(self,
				 #encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
				 input_vocabulary_dim, target_vocabulary_dim, #target_max_length,
				 pad_symbol_idx, go_symbol_idx, eos_symbol_idx,
				 embedding_dim,
				 embedding_matrix_encoder=None, embedding_matrix_decoder=None,
				 embedding_padding_idx=None):
		# hparams:
		bidirectional = False
		n_layers = 1
		
		#encoder_input_size = input_vocabulary_dim
		encoder_hidden_size = 300
		decoder_hidden_size = 300
		#decoder_output_size = target_vocabulary_dim
		
		#self.target_max_length = target_max_length
		
		self.PAD_SYMBOL_IDX = pad_symbol_idx
		self.GO_SYMBOL_IDX = go_symbol_idx
		self.EOS_SYMBOL_IDX = eos_symbol_idx
	
		# Encoder:
		self.encoder = Encoder.Encoder(input_vocabulary_dim,
									   embedding_dim,
									   encoder_hidden_size,
									   n_layers,
									   bidirectional,
									   embedding_matrix_encoder,
									   embedding_padding_idx)
									   
		# Decoder:
		self.decoder = Decoder.Decoder(target_vocabulary_dim,
									   embedding_dim,
									   decoder_hidden_size,
									   n_layers,
									   bidirectional,
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

	def train(self, bucket_list_X, bucket_list_Y, batch_size=32, epochs=10, validation_data=None):
		
		# Check if dev set is defined:
		if validation_data is not None:
			dev_bucket_list_X = validation_data[0]
			dev_bucket_list_Y = validation_data[1]
		
		# Used to count tot. number of iterations:
		tot_sentences = sum([len(b) for b in bucket_list_X])
		
		for epoch in range(epochs):
		
			print("Epoch " + str(epoch+1) + "/" + str(epochs))
			print("----------")
		
			# Used to compute accuracy over buckets:
			correct_predicted_words_cnt = 0
			words_train_cnt = 0
			
			# Used to compute loss over buckets
			tot_loss = 0
			sentences_train_cnt = 0
		
			for X, Y in zip(bucket_list_X, bucket_list_Y):
				for idx in range(0, len(X), batch_size):
				
					# Init tensors:
					x = torch.autograd.Variable(torch.LongTensor(X[idx:min(idx+batch_size, len(X))]))
					y = torch.autograd.Variable(torch.LongTensor(Y[idx:min(idx+batch_size, len(Y))]))
					
					if torch.cuda.is_available():
						x = x.cuda()
						y = y.cuda()
					
					self.encoder_optimizer.zero_grad()
					self.decoder_optimizer.zero_grad()

					#target_length = y.size()[1]
					sentences_train_cnt += x.size()[0]

					#encoder_outputs = torch.autograd.Variable(torch.zeros(self.target_max_length, self.encoder.hidden_size))
					#encoder_outputs = encoder_outputs.cuda() if torch.cuda.is_available() else encoder_outputs
	
					loss, correct_predicted_words_c, words_train_c = self._forward(x, y)
					tot_loss += loss.data[0]
					correct_predicted_words_cnt += correct_predicted_words_c
					words_train_cnt += words_train_c

					# Print current status of training:
					print("Iter: " + str(sentences_train_cnt/tot_sentences*100) + "%" +
						  " | Training Loss: " + str(tot_loss/sentences_train_cnt) +
						  " | Training Accuracy: " + str(correct_predicted_words_cnt/words_train_cnt*100) + "%", end="\r")

					# Compute gradients:
					loss.backward()

					# Update the parameters of the network:
					self.encoder_optimizer.step()
					self.decoder_optimizer.step()

			print("")
			if validation_data is not None:
				validation_loss, validation_accuracy = self.evaluate(dev_bucket_list_X, dev_bucket_list_Y, batch_size)
				print("Validation Loss: " + str(validation_loss) + " | Validation Accuracy: " + str(validation_accuracy*100))
			print("")

	# Evaluate the network on a list of buckets,
	# returns loss and accuracy:
	def evaluate(self, bucket_list_X, bucket_list_Y, batch_size=32):
	
		# Used to count tot. number of iterations:
		tot_sentences = sum([len(b) for b in bucket_list_X])
	
		# Used to compute accuracy over buckets:
		correct_predicted_words_cnt = 0
		words_cnt = 0
		
		# Used to compute loss over buckets
		tot_loss = 0
		sentences_cnt = 0
	
		for X, Y in zip(bucket_list_X, bucket_list_Y):
			for idx in range(0, len(X), batch_size):
				# Init tensors:
				x = torch.autograd.Variable(torch.LongTensor(X[idx:min(idx+batch_size, len(X))]))
				y = torch.autograd.Variable(torch.LongTensor(Y[idx:min(idx+batch_size, len(Y))]))
				
				if torch.cuda.is_available():
					x = x.cuda()
					y = y.cuda()

				sentences_cnt += x.size()[0]
	
				loss, correct_predicted_words_c, words_c = self._forward(x, y)
				tot_loss += loss.data[0]
				correct_predicted_words_cnt += correct_predicted_words_c
				words_cnt += words_c
	
				print("Iter: " + str(sentences_cnt/tot_sentences*100) + "%", end="\r")
	
		return tot_loss / sentences_cnt, correct_predicted_words_cnt / words_cnt

	# x and y are tensors of shape (batch_size, seq_len).
	# It performs a forward step of the whole seq2seq network computing
	# the loss, the tot. number of correct predicted words and the
	# total number of words (removes padding from counting):
	def _forward(self, x, y):
	
		correct_predicted_words_cnt = 0
		words_cnt = 0
	
		target_length = y.size()[1]
		loss = 0

		encoder_output, encoder_hidden = self.encoder(x)

		decoder_input = torch.autograd.Variable(torch.LongTensor([[self.GO_SYMBOL_IDX] * x.size()[0]]))
		decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

		decoder_hidden = encoder_hidden

		for di in range(target_length):
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
			topv, topi = decoder_output.data.topk(1)

			decoder_input = torch.autograd.Variable(torch.LongTensor(topi))
			decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input

			loss += self.criterion(decoder_output, y[:,di])

			# Compute number of correct predictions over current batch:
			for word_pred, word_true in zip(topi, y[:,di]):
				# NOTE: word_pred is a torch Tensor, word_true is a torch Variable.
				word_pred = word_pred[0]
				word_true = word_true.data[0]
				if word_true != self.PAD_SYMBOL_IDX:
					words_cnt += 1
					if word_pred == word_true:
						correct_predicted_words_cnt += 1

		return loss, correct_predicted_words_cnt, words_cnt
