import torch

def train(model, optimizer, criterion, bucket_list_X, bucket_list_Y, batch_size=32, epochs=10, validation_data=None):
	
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
				
				optimizer.zero_grad()

				#target_length = y.size()[1]
				sentences_train_cnt += x.size()[0]

				#encoder_outputs = torch.autograd.Variable(torch.zeros(self.target_max_length, self.encoder.hidden_size))
				#encoder_outputs = encoder_outputs.cuda() if torch.cuda.is_available() else encoder_outputs

				loss, correct_predicted_words_c, words_train_c = _compute_loss(model, criterion, x, y)
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
				optimizer.step()

		print("")
		if validation_data is not None:
			validation_loss, validation_accuracy = evaluate(model, criterion, dev_bucket_list_X, dev_bucket_list_Y, batch_size)
			print("Validation Loss: " + str(validation_loss) + " | Validation Accuracy: " + str(validation_accuracy*100))
		print("")

# Evaluate the network on a list of buckets,
# returns loss and accuracy:
def evaluate(model, criterion, bucket_list_X, bucket_list_Y, batch_size=32):

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

			loss, correct_predicted_words_c, words_c = _compute_loss(model, criterion, x, y)
			tot_loss += loss.data[0]
			correct_predicted_words_cnt += correct_predicted_words_c
			words_cnt += words_c

			print("Iter: " + str(sentences_cnt/tot_sentences*100) + "%", end="\r")

	return tot_loss / sentences_cnt, correct_predicted_words_cnt / words_cnt

# x and y are tensors of shape (batch_size, seq_len).
# It performs a forward step of the whole seq2seq network computing
# the loss, the tot. number of correct predicted words and the
# total number of words (removes padding from counting):
def _compute_loss(model, criterion, x, y):

	encoder_input = x
	decoder_input = torch.autograd.Variable(torch.LongTensor([[model.GO_SYMBOL_IDX] * x.size()[0]]))
	decoder_input = decoder_input.cuda() if torch.cuda.is_available() else decoder_input
	target_length = y.size()[1]
	list_y_p = model(encoder_input, decoder_input, target_length)

	correct_predicted_words_cnt = 0
	words_cnt = 0
	loss = 0

	for di, y_p in enumerate(list_y_p):
		topv, topi = y_p.data.topk(1)
		loss += criterion(y_p, y[:,di])
	
		# Compute number of correct predictions over current batch:
		for word_pred, word_true in zip(topi, y[:,di]):
			# NOTE: word_pred is a torch Tensor, word_true is a torch Variable.
			word_pred = word_pred[0]
			word_true = word_true.data[0]
			if word_true != model.PAD_SYMBOL_IDX:
				words_cnt += 1
				if word_pred == word_true:
					correct_predicted_words_cnt += 1

	return loss, correct_predicted_words_cnt, words_cnt