import random
import utils

class QuestionGenerator:
	def __init__(self, babeldomains_file_path, domains_to_relations_file_path, questionPatterns):
		
		self.domain_to_concepts = {}
		self.domain_to_relations = {}
		self.questionPatterns = questionPatterns
		
		# Open concept to domain file:
		with open(babeldomains_file_path) as babeldomains_file:
			print("Reading", babeldomains_file_path)
			lines = babeldomains_file.readlines()
			cnt = 0
			for l in lines:
				cnt += 1
				print("Progress: {:2.1%}".format(cnt / len(lines)), end="\r")
				
				l = l.rstrip("\n").split("\t")
				
				if l[1] not in self.domain_to_concepts:
					self.domain_to_concepts[l[1]] = []
				
				self.domain_to_concepts[l[1]].append(l[0])
			print("\nDone.")

		# Open domain to relation file:
		with open(domains_to_relations_file_path) as domains_to_relations_file:
			for l in domains_to_relations_file:
				l = l.rstrip("\n").split("\t")
				self.domain_to_relations[l[0]] = l[1:]

	# Generate a random question given a domain:
	def generate(self, domain, babelNetCache=None):
		question_data = {}
		
		id1 = random.choice(self.domain_to_concepts[domain])
		w1 = utils.babelNetIdToLemma(id1, babelNetCache)
		
		relation = random.choice(self.domain_to_relations[domain])
		question = random.choice(self.questionPatterns[relation])

		if "X" in question and "Y" in question:
			question_data["id1"] = id1
			question_data["c1"] = w1
			
			id2 = random.choice(self.domain_to_concepts[domain])
			question_data["id2"] = id2
			
			w2 = babelNetIdToLemma(id2)
			question_data["c2"] = w2
			
			question = question.replace("X", w1)
			question = question.replace("Y", w2)
		
			question_data["type"] = "XY"
		elif "X" in question:
			question_data["id1"] = id1
			question_data["c1"] = w1
			
			question = question.replace("X", w1)
			question_data["type"] = "X"
		else:
			question_data["id2"] = id1
			question_data["c2"] = w1
			
			question = question.replace("Y", w1)
			question_data["type"] = "Y"

		question_data["relation"] = relation
		question_data["question"] = question

		return question_data
