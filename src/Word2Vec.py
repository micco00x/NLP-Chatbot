from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import utils
from Vocabulary import Vocabulary

# Word2Vec class which supports word2vec model from Google with embedding dim. of 300:
class Word2Vec:
	def __init__(self, word2vec_path):#, vocabulary_path):
		
		# Read word2vec binary file:
		self.vecmodel = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		self.EMBEDDING_DIM = 300
		
		# Set up vocabulary:
		#self.vocabulary = Vocabulary(vocabulary_path)
		
		# Set up embedding_matrix:
		#self.embedding_matrix = np.zeros((self.vocabulary.VOCABULARY_DIM, self.EMBEDDING_DIM))
		
		#for word, idx in vocabulary.word2index.items():
		#	try:
		#		self.embedding_matrix[idx] = vecmodel[word]
		#	except KeyError:
		#		self.embedding_matrix[idx] = [0] * self.EMBEDDING_DIM

	# Creates and embedding matrix given a Vocabulary object:
	def createEmbeddingMatrix(self, vocabulary):
		embedding_matrix = np.zeros((vocabulary.VOCABULARY_DIM, self.EMBEDDING_DIM))
		
		for word, idx in vocabulary.word2index.items():
			try:
				embedding_matrix[idx] = self.vecmodel[word]
			except KeyError:
				embedding_matrix[idx] = [0] * self.EMBEDDING_DIM

		return embedding_matrix
