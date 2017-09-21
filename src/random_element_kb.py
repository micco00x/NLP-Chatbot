import json
import random

# Open the Knowledge Base:
print("Loading the knowledge base...")
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)
print("Done.")

while True:
	cmd = input("> ")
	if cmd == "q":
		break
	else:
		print(random.choice(knowledge_base))
