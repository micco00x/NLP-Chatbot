import json
import sys
from utils import split_words_punctuation


# Open the Knowledge Base:
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)

# Number of minimum counts to include a word
# in the vocabulary:
THRESHOLD = int(sys.argv[1])

cnt = 0
kb_len = len(knowledge_base)
print("Reading the knowledge base (" + str(kb_len) + " elements)")

words = {}
vocabulary = []

for elem in knowledge_base:

	cnt += 1
	print("Progress: {:2.1%}".format(cnt / kb_len), end="\r")

	question = elem["question"]
	answer = elem["answer"]
	
	for w in split_words_punctuation(elem["question"]):
		if w not in words:
			words[w] = 0
		words[w] += 1
	
	for w in split_words_punctuation(elem["answer"]):
		if w not in words:
			words[w] = 0
		words[w] += 1

print("\nDone.")

M = 20
m = [0] * M

for key, value in words.items():
	#print (key, value)
	for i in range(M):
		if value >= i:
			m[i] += 1
	if value >= THRESHOLD:
		vocabulary.append(key)

for i in range(M):
	print("m" + str(i) + ":", m[i])

vocabulary = sorted(vocabulary)

vocabulary_file = open("../resources/vocabulary.txt", "w")
for w in vocabulary:
	vocabulary_file.write(w + "\n")
vocabulary_file.close()

print("Vocabulary of", len(vocabulary), "words created.")
