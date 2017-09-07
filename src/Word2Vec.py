from gensim.models.keyedvectors import KeyedVectors

import numpy as np

from utils import split_words_punctuation

# Word2Vec class which supports word2vec model from Google with embedding dim. of 300:
class Word2Vec:
	def __init__(self, word2vec_path, vocabulary_path):
		
		vecmodel = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
		self.EMBEDDING_DIM = 300
		
		vocabulary = []
		with open(vocabulary_path) as vocabulary_file:
			for w in vocabulary_file:
				vocabulary.append(w.rstrip("\n"))
	
		veclist = []
		wordlist = []
		
		for word in vocabulary:
			try:
				if word not in wordlist:
					wordvec = vecmodel[word]
					veclist.append(wordvec)
					wordlist.append(word)
			except KeyError:
				# TO TEST:
				wordvec = [0] * self.EMBEDDING_DIM
				veclist.append(wordvec)
				wordlist.append(word)
				#pass

		self.UNK_SYMBOL = 0
		self.GO_SYMBOL = 1
		self.EOS_SYMBOL = 2
		
		self.VOCABULARY_DIM = len(wordlist) + 3
		
		self.word_to_indices = dict((c, i) for i, c in enumerate(wordlist))
		#self.indices_to_word = dict((i, c) for i, c in enumerate(wordlist))

		self.embedding_matrix = np.zeros((self.VOCABULARY_DIM, self.EMBEDDING_DIM))
		
		for i in range(len(wordlist)):
			self.embedding_matrix[i + 3] = veclist[i]

	# Word to index (+1 because of OOV word):
	def w2i(self, word):
		if word in self.word_to_indices:
			return self.word_to_indices[word] + 3
		else:
			return 0

	# Sentence to word embeddings (indices):
	def s2i(self, sentence):
		q = []
		for w in split_words_punctuation(sentence):
			q.append(self.w2i(w))
		return q
