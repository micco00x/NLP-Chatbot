import json

from BabelNetCache import *
from utils import *
from Vocabulary import Vocabulary
from Word2Vec import Word2Vec

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils import np_utils

import torch
from seq2seq.Seq2Seq import Seq2Seq
import seq2seq.utils

import numpy as np

# TODO: create a function split_dataset(X, Y), returns X_train, Y_train, X_dev, Y_dev, X_test, Y_test

# Create buckets with padding given two sets and the dimensions of the buckets,
# returns a list of buckets for X and a list of buckets for Y:
def create_buckets(bucket_list_X, bucket_list_Y, dims):
	bucket_x = [[] for _ in range(len(dims))]
	bucket_y = [[] for _ in range(len(dims))]
	padded_bucket_x = [[] for _ in range(len(dims))]
	padded_bucket_y = [[] for _ in range(len(dims))]

	# Put elements of X, Y in buckets depending
	# on the length of the target sentence:
	for x, y in zip(bucket_list_X, bucket_list_Y):
		for idx, d in enumerate(dims):
			if len(y) <= d:
				bucket_x[idx].append(x)
				bucket_y[idx].append(y)

	# Add padding to buckets:
	for idx, (bx, by) in enumerate(zip(bucket_x, bucket_y)):
		max_len_x = max([len(x) for x in bx])
		max_len_y = max([len(y) for y in by])
		for x, y in zip(bx, by):
			x[-1:-1] = [0] * (max_len_x - len(x))
			y[-1:-1] = [0] * (max_len_y - len(y))
			x.reverse() # use reversed sentence to increase amount of short term dependencies
			padded_bucket_x[idx].append(x)
			padded_bucket_y[idx].append(y)

	return padded_bucket_x, padded_bucket_y

# Models to train:
TRAIN_RELATION_CLASSIFIER = False
TRAIN_CONCEPT_EXTRACTOR = False
TRAIN_ANSWER_GENERATOR = True

# Open the Knowledge Base:
print("Loading the knowledge base...")
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)
print("Done.")

# Vocabulary and Word2Vec:
vocabulary_small = Vocabulary("../resources/vocabulary_138K.txt")
vocabulary_big = Vocabulary("../resources/vocabulary_138K.txt")
print("Loading Word2Vec...")
word2vec = Word2Vec("../resources/Word2Vec.bin")
print("Done.")

##### RELATION CLASSIFIER #####
if TRAIN_RELATION_CLASSIFIER == True:
	
	print("Training relation classifier")

	X       = []
	Y       = []
	X_train = []
	Y_train = []
	X_dev   = []
	Y_dev   = []
	X_test  = []
	Y_test  = []

	cnt = 0
	kb_len = len(knowledge_base)
	print("Reading the knowledge base (" + str(kb_len) + " elements)")

	# Build X and Y:
	for elem in knowledge_base: # TODO: TRAIN ON FIRST N ELEMENTS OF THE KB (CPU IS SLOW)

		cnt += 1
		print("Progress: {:2.1%}".format(cnt / kb_len), end="\r")

		X.append(vocabulary_big.sentence2indices(elem["question"]))
		Y.append(relation_to_int(elem["relation"]))

	print("\nDone.")

	# Relation to one hot enconding:
	Y = keras.utils.np_utils.to_categorical(Y, 16)

	# Add padding to X:
	longest_sentence_length = max([len(sentence) for sentence in X])
	X = keras.preprocessing.sequence.pad_sequences(sequences=X, maxlen=longest_sentence_length)

	# Split training set into train, dev and test:
	KB_SPLIT = 0.6
	X_train = np.array(X[:int(len(X) * KB_SPLIT)])
	Y_train = np.array(Y[:int(len(Y) * KB_SPLIT)])
	X_dev   = np.array(X[int(len(X) * KB_SPLIT):int(len(X) * (KB_SPLIT + 1) / 2)])
	Y_dev   = np.array(Y[int(len(Y) * KB_SPLIT):int(len(Y) * (KB_SPLIT + 1) / 2)])
	X_test  = np.array(X[int(len(X) * (KB_SPLIT + 1) / 2):])
	Y_test  = np.array(Y[int(len(Y) * (KB_SPLIT + 1) / 2):])

	# Define the network:
	relation_classifier = Sequential()
	relation_classifier.add(Embedding(input_dim=vocabulary_big.VOCABULARY_DIM,
									  output_dim=word2vec.EMBEDDING_DIM,
									  weights=[word2vec.createEmbeddingMatrix(vocabulary_big)],
									  trainable=True,
									  mask_zero=True))
	relation_classifier.add(LSTM(units=200, return_sequences=False))
	relation_classifier.add(Dense(16))
	relation_classifier.add(Activation("softmax"))

	# Compile the network:
	relation_classifier.compile(loss="categorical_crossentropy",
								optimizer=RMSprop(lr=0.01),
								metrics=["accuracy"])

	# Train the network:
	relation_classifier.fit(X_train, Y_train,
							validation_data=(X_dev, Y_dev),
							batch_size=128,
							epochs=5)

	# Results of the network on the test set:
	loss_and_metrics = relation_classifier.evaluate(X_test, Y_test)
	print(relation_classifier.metrics_names[0] + ": " + str(loss_and_metrics[0]))
	print(relation_classifier.metrics_names[1] + ": " + str(loss_and_metrics[1]))

	# Save the network:
	relation_classifier.save("../models/relation_classifier.keras")

##### CONCEPT EXTRACTOR #####
if TRAIN_CONCEPT_EXTRACTOR == True:
	print("Training concept extractor")
	
	babelNetCache = BabelNetCache("../resources/babelnet_cache.tsv")

	X = []
	Y = []

	cnt = 0
	kb_len = len(knowledge_base)
	print("Reading the knowledge base (" + str(kb_len) + " elements)")
	
	for elem in knowledge_base:
		
		cnt += 1
		print("Progress: {:2.1%}".format(cnt / kb_len), end="\r")
		
		question = elem["question"].strip().rstrip()
		answer = elem["answer"].strip().rstrip()
		c2 = elem["c2"].strip().rstrip()
		#print("Q:", question)
		#print("A:", answer)
		#print("c2:", c2)
		if answer.lower() == "yes" or answer.lower() == "no":
			# c1 and c2 can be determined directly inside the question
			#print("c1 and c2 can be determined directly inside the question")
			continue
		elif c2.count("bn:") >= 2:
			# c2 is malformed
			#print("c2 is malformed")
			continue
		elif "::bn:" in c2: # case "w::bn:--n"
			i = c2.index("::bn:")
			w = c2[:i].strip().rstrip()
			
			answer_split = split_words_punctuation(answer)
			w_split = split_words_punctuation(w)
			
			i1 = find_pattern(answer_split, w_split)
			i2 = i1 + len(w_split) - 1
		elif "bn:" in c2: # case "bn:--n"
			try:
				# TODO: note that using regex could help finding "bn:--n" better
				#print("Case bn:--n")
				#print(c2[c2.index("bn:"):])
				w = babelNetIdToLemma(c2[c2.index("bn:"):], babelNetCache)
				#print("w:", w)
				
				answer_split = split_words_punctuation(answer.lower())
				w_split = split_words_punctuation(w.lower())
				
				# TODO: note that len(answer_split) could be less than len(w_split)
				i1 = find_pattern(answer_split, w_split)
				i2 = i1 + len(w_split) - 1
			except Exception as e:
				#print(str(e))
				continue
		elif c2.lower() in answer.lower(): # case "w"
			answer_split = split_words_punctuation(answer)
			c2_split = split_words_punctuation(c2)
		
			i1 = find_pattern(answer_split, c2_split)
			i2 = i1 + len(c2_split) - 1
		else:
			continue

		# Create data for the NN:
		x = vocabulary_big.sentence2indices(answer)
		y = [[0, 0, 0, 1] for _ in range(len(x))]

		if i1 == -1 or i2 == -1:
			#print("ERROR: index -1")
			continue

		# The KB could be malformed, validate i1 and i2:
		i1 = max(i1, 0)
		i2 = min(i2, len(x)-1)

		#print("i1:", i1)
		#print("i2:", i2)
		
		# Begin and end of the concept,
		# activation is (Begin+End, Begin (but not End), End (but not Begin), Other (not Begin nor End):
		if i1 == i2:
			y[i1] = [1, 0, 0, 0]
		else:
			y[i1] = [0, 1, 0, 0]
			y[i2] = [0, 0, 1, 0]
		
		#print("x:", x)
		#print("y:", y)
	
		X.append(x)
		Y.append(y)

	print("\nDone.")

	# Save the cache with the new elements found from the queries:
	babelNetCache.save()

	# Add padding to X and Y:
	longest_sentence_length = max([len(sentence) for sentence in X])
	X = keras.preprocessing.sequence.pad_sequences(sequences=X, maxlen=longest_sentence_length)
	Y = keras.preprocessing.sequence.pad_sequences(sequences=Y, maxlen=longest_sentence_length)

	# Split training set into train, dev and test:
	KB_SPLIT = 0.6
	X_train = np.array(X[:int(len(X) * KB_SPLIT)])
	Y_train = np.array(Y[:int(len(Y) * KB_SPLIT)])
	X_dev   = np.array(X[int(len(X) * KB_SPLIT):int(len(X) * (KB_SPLIT + 1) / 2)])
	Y_dev   = np.array(Y[int(len(Y) * KB_SPLIT):int(len(Y) * (KB_SPLIT + 1) / 2)])
	X_test  = np.array(X[int(len(X) * (KB_SPLIT + 1) / 2):])
	Y_test  = np.array(Y[int(len(Y) * (KB_SPLIT + 1) / 2):])

	# Define the network:
	concept_extractor = Sequential()
	concept_extractor.add(Embedding(input_dim=vocabulary_big.VOCABULARY_DIM,
									output_dim=word2vec.EMBEDDING_DIM,
									weights=[word2vec.createEmbeddingMatrix(vocabulary_big)],
									trainable=True,
									mask_zero=True))
	concept_extractor.add(LSTM(units=200, return_sequences=True))
	concept_extractor.add(Dense(4))
	concept_extractor.add(Activation("softmax"))

	# Compile the network:
	concept_extractor.compile(loss="categorical_crossentropy",
							  optimizer=RMSprop(lr=0.01),
							  metrics=["accuracy"])
	
	# Train the network:
	concept_extractor.fit(X_train, Y_train,
						  validation_data=(X_dev, Y_dev),
						  batch_size=128,
						  epochs=5)

	# Results of the network on the test set:
	loss_and_metrics = concept_extractor.evaluate(X_test, Y_test)
	print(concept_extractor.metrics_names[0] + ": " + str(loss_and_metrics[0]))
	print(concept_extractor.metrics_names[1] + ": " + str(loss_and_metrics[1]))
	
	# Save the network:
	concept_extractor.save("../models/concept_extractor.keras")

if TRAIN_ANSWER_GENERATOR == True:
	print("Training answer generator")

	X = []
	Y = []
	
	cnt = 0
	kb_len = len(knowledge_base)
	print("Reading the knowledge base (" + str(kb_len) + " elements)")
	
	for elem in knowledge_base[:5000]:
		cnt += 1
		print("Progress: {:2.1%}".format(cnt / kb_len), end="\r")
		
		question = elem["question"].strip().rstrip()
		answer = elem["answer"].strip().rstrip()

		x = vocabulary_big.sentence2indices(question)
		x.append(vocabulary_big.word2index[vocabulary_big.EOS_SYMBOL])
		y = vocabulary_small.sentence2indices(answer)
		y.append(vocabulary_small.word2index[vocabulary_small.EOS_SYMBOL])
	
		X.append(x)
		Y.append(y)
	
	print("\nDone.")
	
	# Split training set into train, dev and test:
	KB_SPLIT = 0.6
	X_train = X[:int(len(X) * KB_SPLIT)]
	Y_train = Y[:int(len(Y) * KB_SPLIT)]
	X_dev   = X[int(len(X) * KB_SPLIT):int(len(X) * (KB_SPLIT + 1) / 2)]
	Y_dev   = Y[int(len(Y) * KB_SPLIT):int(len(Y) * (KB_SPLIT + 1) / 2)]
	X_test  = X[int(len(X) * (KB_SPLIT + 1) / 2):]
	Y_test  = Y[int(len(Y) * (KB_SPLIT + 1) / 2):]
	
	# Create buckets:
	buckets_dims = [10, 20, 50, max([len(y) for y in Y])]
	padded_bucket_x_train, padded_bucket_y_train = create_buckets(X_train, Y_train, buckets_dims)
	padded_bucket_x_dev, padded_bucket_y_dev = create_buckets(X_dev, Y_dev, buckets_dims)
	padded_bucket_x_test, padded_bucket_y_test = create_buckets(X_test, Y_test, buckets_dims)

	# Define the network:
	emb_matrix_big = word2vec.createEmbeddingMatrix(vocabulary_big)
	emb_matrix_small = word2vec.createEmbeddingMatrix(vocabulary_small)
	seq2seq_model = Seq2Seq(vocabulary_big.VOCABULARY_DIM, vocabulary_small.VOCABULARY_DIM,
							vocabulary_small.word2index[vocabulary_small.PAD_SYMBOL],
							vocabulary_small.word2index[vocabulary_small.GO_SYMBOL],
							vocabulary_small.word2index[vocabulary_small.EOS_SYMBOL],
							300, 300,
							word2vec.EMBEDDING_DIM, emb_matrix_big, emb_matrix_small,
							vocabulary_small.word2index[vocabulary_small.PAD_SYMBOL])
	seq2seq_model = seq2seq_model.cuda() if torch.cuda.is_available() else seq2seq_model

	# Train the network:
	optimizer = torch.optim.RMSprop(seq2seq_model.parameters())
	criterion = torch.nn.NLLLoss(ignore_index=seq2seq_model.embedding_padding_idx)
	batch_size = 128
	seq2seq.utils.train(seq2seq_model,
					    optimizer,
						criterion,
						padded_bucket_x_train, padded_bucket_y_train,
						batch_size=batch_size, epochs=5,
						validation_data=[padded_bucket_x_dev, padded_bucket_y_dev])

	# Test the network:
	seq2seq.utils.evaluate(seq2seq_model, criterion, padded_bucket_x_test, padded_bucket_y_test, batch_size)
