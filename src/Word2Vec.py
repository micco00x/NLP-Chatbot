from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import utils
from Vocabulary import Vocabulary

# Word2Vec class which supports word2vec model from Google with embedding dim. of 300:
class Word2Vec:
	def __init__(self, word2vec_path, vocabulary_path):
		
		# Read word2vec binary file:
		vecmodel = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		self.EMBEDDING_DIM = 300
		
		# Set up vocabulary:
		self.vocabulary = Vocabulary(vocabulary_path)
		
		# Set up embedding_matrix:
		self.embedding_matrix = np.zeros((self.vocabulary.VOCABULARY_DIM, self.EMBEDDING_DIM))
		
		for word, idx in vocabulary.word2index.items():
			try:
				self.embedding_matrix[idx] = vecmodel[word]
			except KeyError:
				self.embedding_matrix[idx] = [0] * self.EMBEDDING_DIM

	# Word to index (+1 because of OOV word):
	def w2i(self, word):
		if word in self.vocabulary.word2index:
			return self.vocabulary.word2index[word]
		else:
			return 0

	# Sentence to indices:
	def s2i(self, sentence):
		q = []
		for w in utils.split_words_punctuation(sentence):
			q.append(self.w2i(w))
		return q
