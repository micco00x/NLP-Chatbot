import json
from utils import split_words_punctuation

# Open the Knowledge Base:
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)

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

m2 = 0
m3 = 0
m4 = 0
THRESHOLD = 2

for key, value in words.items():
	#print (key, value)
	if value >= 2:
		m2 += 1
	if value >= 3:
		m3 += 1
	if value >= 4:
		m4 += 1
	if value >= THRESHOLD:
		vocabulary.append(key)

print("m1:", len(words))
print("m2:", m2)
print("m3:", m3)
print("m4:", m4)

vocabulary = sorted(vocabulary)

vocabulary_file = open("../resources/vocabulary.txt", "w")
for w in vocabulary:
	vocabulary_file.write(w + "\n")
vocabulary_file.close()

print("Vocabulary of", len(vocabulary), "words created.")
