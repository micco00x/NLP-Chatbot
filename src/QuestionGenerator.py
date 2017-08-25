import urllib.request
import json
import gzip

from io import BytesIO

import random

from utils import *

# Open the Knowledge Base:
#with open("../knowledge_base/kb.json") as kb_file:
#	knowledge_base = json.load(kb_file)

#for elem in knowledge_base:
#	if len(elem["domains"]) != 0:
#		if elem["domains"][0] != "":
#			print(elem["domains"])

class QuestionGenerator:
	def __init__(self, babeldomains_file_path, domains_to_relations_file_path, patterns_file_path):
		
		self.domain_to_concepts = {}
		self.domain_to_relations = {}
		self.relation_to_questions = {}
		
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

		# Open patterns file:
		with open(patterns_file_path) as patterns_file:
			for l in patterns_file:
				l = l.rstrip("\n").split("\t")
				
				if l[1] not in self.relation_to_questions:
					self.relation_to_questions[l[1]] = []
				
				self.relation_to_questions[l[1]].append(l[0])

	# Generate a random question given a domain:
	def generate(self, domain, babelNetCache=None):
		question_data = {}
		
		id1 = random.choice(self.domain_to_concepts[domain])
		#question_data["id1"] = id1
		
		# TO TEST: add babelnetcache here (speeds up question generation)
		w1 = babelNetIdToLemma(id1, babelNetCache)
		
		relation = random.choice(self.domain_to_relations[domain])
		question = random.choice(self.relation_to_questions[relation])

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

#domains = []

# Open domains file:
#with open("../babelnet/BabelDomains_full/domain_list.txt") as domain_list_file:
#	for l in domain_list_file:
#		domains.append(l.rstrip("\n"))

#question_generator = QuestionGenerator("../babelnet/BabelDomains_full/BabelDomains/babeldomains_babelnet.txt",
#									   "../resources/domains_to_relations.tsv",
#									   "../resources/patterns.tsv")

#chosen = False
#while not chosen:
#	try:
#		question_data = question_generator.generate(random.choice(domains))
#		chosen = True
#	except:
#		pass

#print(question_data)

