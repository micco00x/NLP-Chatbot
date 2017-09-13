import utils

# Vocabulary class which support UNK, GO, EOS symbols
# and a list of words from a vocabulary file:
class Vocabulary:
	def __init__(self, vocabulary_path=None):

		# Attributes of the class:
		self.VOCABULARY_DIM = 0
		self.word2index = {}
		self.index2word = {}
		
		# Special symbols:
		self.PAD_SYMBOL = "<PAD>"
		self.UNK_SYMBOL = "<UNK>"
		self.GO_SYMBOL = "<GO>"
		self.EOS_SYMBOL = "<EOS>"

		# Add special symbols to Vocabulary:
		self.addWord(self.PAD_SYMBOL)
		self.addWord(self.UNK_SYMBOL)
		self.addWord(self.GO_SYMBOL)
		self.addWord(self.EOS_SYMBOL)

		# Add words from vocabulary path if specified:
		if vocabulary_path is not None:
			with open(vocabulary_path) as vocabulary_file:
				for w in vocabulary_file:
					self.addWord(w.rstrip("\n"))

	# Add word to vocabulary:
	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.VOCABULARY_DIM
			self.index2word[self.VOCABULARY_DIM] = word
			self.VOCABULARY_DIM += 1

	# Return a list of indices given a sentence:
	def sentence2indices(self, sentence):
		q = []
		for w in utils.split_words_punctuation(sentence):
			if w in self.word2index:
				q.append(self.word2index[w])
			else:
				q.append(0)
		return q
