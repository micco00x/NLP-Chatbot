import json

class KnowledgeBase:
	def __init__(self, kb_path):
		# Open the Knowledge Base:
		print("Loading the knowledge base...")
		with open(kb_path) as kb_file:
			self.kb = json.load(kb_file)
		print("Done.")

	# Search for the element in the KB that best matches the given relation, c1 and c2:
	def search(self, babelNetCache, relation, concept1=None, concept2=None):
		if concept1: concept1 = concept1.lower()
		if concept2: concept2 = concept2.lower()
		print("Searching in the KB:")
		print("\t" + relation + "\t" + str(concept1) + "\t" + str(concept2))
		for elem in self.kb:
			if elem["relation"] == relation:
				c1_kb = self._concept_to_word(babelNetCache, elem["c1"])
				c2_kb = self._concept_to_word(babelNetCache, elem["c2"])
				if concept1 != None and concept2 != None and c1_kb != None and c2_kb != None:
					if concept1 in c1_kb and concept2 in c2_kb:
						return elem
				elif concept1 != None and c1_kb != None:
					if concept1 in c1_kb:
						return elem
				elif concept2 != None and c2_kb != None:
					if concept2 in c2_kb:
						return elem
		return None


	# Transform the concept of the KB to the corresponding word:
	def _concept_to_word(self, babelNetCache, concept):
		if concept.count("bn:") > 2:
			# Malformed tuple:
			return None
		elif "::" in concept:
			# Case w::bnid:
			idx = concept.index("::")
			w = concept[:idx].lower()
			return w
		elif "bn:" in concept:
			# Case bnid:
			try:
				return babelNetCache.cache[concept[concept.index("bn:"):]].lower()
			except:
				pass
		else:
			# Case w:
			return concept.lower()
