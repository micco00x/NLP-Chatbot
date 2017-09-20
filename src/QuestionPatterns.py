class QuestionPatterns:
	def __init__(self, patterns_file_path):
		self.relation_to_questions = {}
	
		# Open patterns file:
		with open(patterns_file_path) as patterns_file:
			for l in patterns_file:
				l = l.rstrip("\n").split("\t")
				
				if l[1] not in self.relation_to_questions:
					self.relation_to_questions[l[1]] = []
				
				self.relation_to_questions[l[1]].append(l[0])

	def __getitem__(self, relation):
		return self.relation_to_questions[relation]
