import torch
import shutil

def train(model, optimizer, criterion,
		  bucket_list_X, bucket_list_Y,
		  batch_size=32, epochs=10,
		  validation_data=None,
		  checkpoint_dir=None,
		  early_stopping_max=None,
		  starting_epoch=0,
		  starting_iter=0,
		  best_acc=0):
	
	# Check if dev set is defined:
	if validation_data is not None:
		dev_bucket_list_X = validation_data[0]
		dev_bucket_list_Y = validation_data[1]
	
	# Used to count tot. number of iterations:
	tot_sentences = sum([len(b) for b in bucket_list_X])

	# Early stopping counter (epochs without improvements):
	early_stopping_cnt = 0
	
	for epoch in range(starting_epoch, epochs):
	
		print("Epoch " + str(epoch+1) + "/" + str(epochs))
		print("----------")
	
		# Count number of iterations:
		iter_cnt = 0
	
		# Used to compute accuracy over buckets:
		correct_predicted_words_cnt = 0
		words_train_cnt = 0
		
		# Used to compute loss over buckets
		tot_loss = 0
		sentences_train_cnt = 0
	
		for X, Y in zip(bucket_list_X, bucket_list_Y):
			for idx in range(0, len(X), batch_size):
			
				# Update iter_cnt:
				iter_cnt += 1
				if iter_cnt < starting_iter:
					continue
			
				# Init tensors:
				x = torch.autograd.Variable(torch.LongTensor(X[idx:min(idx+batch_size, len(X))]), volatile=model.volatile)
				y = torch.autograd.Variable(torch.LongTensor(Y[idx:min(idx+batch_size, len(Y))]), volatile=model.volatile)
				
				if torch.cuda.is_available():
					x = x.cuda()
					y = y.cuda()
				
				optimizer.zero_grad()

				sentences_train_cnt += x.size()[0]

				loss, correct_predicted_words_c, words_train_c = _compute_loss(model, criterion, x, y)
				tot_loss += loss.data[0]
				correct_predicted_words_cnt += correct_predicted_words_c
				words_train_cnt += words_train_c

				# Print current status of training:
				# TODO: if you start from the middle of an epoch the iter value
				#		is wrong because it considers sentences_train_cnt and not
				#		iter_cnt
				training_loss = tot_loss / sentences_train_cnt
				training_accuracy = correct_predicted_words_cnt / words_train_cnt
				print("Iter: %2.3f%% | Training Loss: %2.3f | Training Accuracy: %2.3f%%" %
					  (sentences_train_cnt/tot_sentences*100, training_loss, training_accuracy*100),
					  end="\r")

				# Compute gradients:
				loss.backward()

				# Update the parameters of the network:
				optimizer.step()
	
				# Save model after 1000 iterations:
				if iter_cnt % 1000 == 0 and checkpoint_dir:
					state = {
						"epoch": epoch+1,
						"iter": iter_cnt+1,
						"state_dict": model.state_dict(),
						"best_acc": best_acc,
						"optimizer": optimizer.state_dict()
					}
					
					filename = checkpoint_dir + "/seq2seq_epoch_" + str(epoch+1) + "_iter_" + str(iter_cnt+1) + ".pth.tar"
					save_checkpoint(state, filename)

		print("")

		# NOTE: if the model is too big loss and accuracy on dev set won't
		#		be computed because the memory of the GPU wouldn't be
		#		enough, during testing this part had been commented to
		#		guarantee the end of the training phase.

		# Compute loss and accuracy on dev set:
		if validation_data:
			validation_loss, validation_accuracy = evaluate(model, criterion, dev_bucket_list_X, dev_bucket_list_Y, batch_size)
			print("Validation Loss: %2.3f | Validation Accuracy: %2.3f%%" % (validation_loss, validation_accuracy*100))
			is_best = best_acc < validation_accuracy
			best_acc = max(best_acc, validation_accuracy)
		else:
			is_best = best_acc < training_accuracy
			best_acc = max(best_acc, training_accuracy)

		print("")

		# Save checkpoint if specified:
		if checkpoint_dir:
			state = {
				"epoch": epoch+1,
				"iter": 0,
				"state_dict": model.state_dict(),
				"best_acc": best_acc,
				"optimizer": optimizer.state_dict()
			}
			
			filename = checkpoint_dir + "/seq2seq_epoch_" + str(epoch+1) + ".pth.tar"
			save_checkpoint(state, filename)
			if is_best:
				shutil.copyfile(filename, checkpoint_dir + "/seq2seq_best.pth.tar")

		# Early stopping:
		if early_stopping_max:
			if is_best:
				early_stopping_cnt = 0
			else:
				early_stopping_cnt += 1
			if early_stopping_max < early_stopping_cnt:
				break

		# Reset starting_iter to 0 at the end of the epoch:
		starting_iter = 0


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
			x = torch.autograd.Variable(torch.LongTensor(X[idx:min(idx+batch_size, len(X))]), volatile=model.volatile)
			y = torch.autograd.Variable(torch.LongTensor(Y[idx:min(idx+batch_size, len(Y))]), volatile=model.volatile)
			
			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()

			sentences_cnt += x.size()[0]

			loss, correct_predicted_words_c, words_c = _compute_loss(model, criterion, x, y)
			tot_loss += loss.data[0]
			correct_predicted_words_cnt += correct_predicted_words_c
			words_cnt += words_c

			print("Iter: %2.3f%%" % (sentences_cnt/tot_sentences*100), end="\r")

	return tot_loss / sentences_cnt, correct_predicted_words_cnt / words_cnt

# x and y are tensors of shape (batch_size, seq_len).
# It performs a forward step of the whole seq2seq network computing
# the loss, the tot. number of correct predicted words and the
# total number of words (removes padding from counting):
def _compute_loss(model, criterion, x, y):

	encoder_input = x
	decoder_input = torch.autograd.Variable(torch.LongTensor([[model.GO_SYMBOL_IDX]] * x.size()[0]), volatile=model.volatile)
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

def save_checkpoint(state, filename):
    torch.save(state, filename)
