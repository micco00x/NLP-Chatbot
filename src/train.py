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

# TODO: create a function split_dataset(X, Y), returns X_train, Y_train, X_dev, Y_dev, X_test, Y_test

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
vocabulary = Vocabulary("../resources/vocabulary.txt")
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

		X.append(vocabulary.sentence2indices(elem["question"]))
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
	relation_classifier.add(Embedding(input_dim=vocabulary.VOCABULARY_DIM,
									  output_dim=word2vec.EMBEDDING_DIM,
									  weights=[word2vec.createEmbeddingMatrix(vocabulary)],
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
		x = vocabulary.sentence2indices(answer)
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
	concept_extractor.add(Embedding(input_dim=vocabulary.VOCABULARY_DIM,
									output_dim=word2vec.EMBEDDING_DIM,
									weights=[word2vec.createEmbeddingMatrix(vocabulary)],
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
	

	for elem in knowledge_base[:50000]:
		
		cnt += 1
		print("Progress: {:2.1%}".format(cnt / kb_len), end="\r")
		
		question = elem["question"].strip().rstrip()
		answer = elem["answer"].strip().rstrip()

		x = vocabulary.sentence2indices(question)
		x.append(vocabulary.word2index[vocabulary.EOS_SYMBOL])
		y = vocabulary.sentence2indices(answer)
		y.append(vocabulary.word2index[vocabulary.EOS_SYMBOL])
		X.append(torch.autograd.Variable(torch.LongTensor(x).view(-1, 1)))
		Y.append(torch.autograd.Variable(torch.LongTensor(y).view(-1, 1)))

	# Split training set into train, dev and test:
	KB_SPLIT = 0.6
	X_train = X[:int(len(X) * KB_SPLIT)]
	Y_train = Y[:int(len(Y) * KB_SPLIT)]
	X_dev   = X[int(len(X) * KB_SPLIT):int(len(X) * (KB_SPLIT + 1) / 2)]
	Y_dev   = Y[int(len(Y) * KB_SPLIT):int(len(Y) * (KB_SPLIT + 1) / 2)]
	X_test  = X[int(len(X) * (KB_SPLIT + 1) / 2):]
	Y_test  = Y[int(len(Y) * (KB_SPLIT + 1) / 2):]

	# Define the network:
	emb_matrix = word2vec.createEmbeddingMatrix(vocabulary)
	seq2seq = Seq2Seq(vocabulary.VOCABULARY_DIM, vocabulary.word2index[vocabulary.GO_SYMBOL], vocabulary.word2index[vocabulary.EOS_SYMBOL],
					  word2vec.EMBEDDING_DIM, emb_matrix, emb_matrix)

	# Train the network:
	seq2seq.train(X_train, Y_train, 5)
