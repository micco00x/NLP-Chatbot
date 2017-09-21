import json
import random

# Open the Knowledge Base:
print("Loading the knowledge base...")
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)
print("Done.")

while True:
	cmd = input("> ")
	if cmd == "quit":
		break
	elif cmd.startswith("search="):
		sentence = cmd[len("search="):].lower()
		for elem in knowledge_base:
			if sentence in elem["question"].lower() or sentence in elem["answer"].lower():
				print(elem)
	elif cmd == "random":
		print(random.choice(knowledge_base))
	else:
		print("Unknown command, try:")
		print(" * quit to exit")
		print(" * search=SENTENCE to search a sentence in the kb")
		print(" * random to select a random element of the kb")
