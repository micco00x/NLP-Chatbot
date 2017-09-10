class Vocabulary:
	def __init__(self, vocabulary_path=None):

		# Attributes of the class:
		self.VOCABULARY_DIM = 0
		self.word2index = {}
		self.index2word = {}
		
		# Special symbols:
		self.UNK_SYMBOL = "<UNK>"
		self.GO_SYMBOL = "<GO>"
		self.EOS_SYMBOL = "<EOS>"

		# Add special symbols to Vocabulary:
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
		if word not in word2index:
			word2index[word] = self.VOCABULARY_DIM
			index2word[self.VOCABULARY_DIM] = word
			self.VOCABULARY_DIM += 1
