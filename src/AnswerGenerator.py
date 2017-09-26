import numpy as np
import utils
import sys

class AnswerGenerator:
	def __init__(self,
				 knowledge_base, question_patterns):
		# Linear algorithm seems to work well with 1M elements
		# but could be quicker with dictionaries:
		self.knowledge_base = knowledge_base
	
		self.question_patterns = question_patterns

	def generate(self, question, babelNetCache):
		#min_d = sys.maxsize
		Q = []
		l = []
		for r in self.question_patterns.relation_to_questions:
			for q_p in self.question_patterns[r]:
				d = utils.levenshtein(question, q_p)
				q_p_pos = len(l)
				for k in range(len(l)):
					if d <= l[k]:
						q_p_pos = k
						break
				Q.insert(q_p_pos, q_p)
				l.insert(q_p_pos, d)

		# Consider first best T matches:
		T = len(Q)

		for q in Q[:T]:
			#print(q)
			Xpos = q.find("X")
			Ypos = q.find("Y")
			
			if Xpos != -1 and Ypos != -1:
			
				#print("BOTH_X_Y")
				
				# Case -- X -- Y --?
				if Xpos < Ypos:
					beforeX = q[:Xpos]
					afterX = q[Xpos+1:Ypos]
					afterY = q[Ypos+1:]
				
					conceptX_begin_idx = -1
					pp_afterx = question[Xpos:].find(afterX)
					if pp_afterx == -1:
						continue
					conceptX_end_idx = Xpos + pp_afterx
					conceptY_begin_idx = -1
					conceptY_end_idx = question.find(afterY)
				
					if question.find(beforeX) != -1:
						conceptX_begin_idx = Xpos # = len(beforeX)
					conceptY_begin_idx = conceptX_end_idx + len(afterX)
				
					#print("CONCEPTX_BEGIN_IDX:", conceptX_begin_idx)
					#print("CONCEPTX_END_IDX:", conceptX_end_idx)
					#print("CONCEPTY_BEGIN_IDX:", conceptY_begin_idx)
					#print("CONCEPTY_END_IDX:", conceptY_end_idx)
			
				# Case -- Y -- X --?
				else:
					beforeY = q[:Ypos]
					afterY = q[Ypos+1:Xpos]
					afterX = q[Xpos+1:]
					
					conceptY_begin_idx = -1
					pp_aftery = question[Ypos:].find(afterY)
					if pp_aftery == -1:
						continue
					conceptY_end_idx = Ypos + pp_aftery
					conceptX_begin_idx = -1
					conceptX_end_idx = question.find(afterX)
				
					if question.find(beforeY) != -1:
						conceptY_begin_idx = Ypos # = len(beforeY)
					conceptX_begin_idx = conceptY_end_idx + len(afterY)
				
					#print("CONCEPTY_BEGIN_IDX:", conceptY_begin_idx)
					#print("CONCEPTY_END_IDX:", conceptY_end_idx)
					#print("CONCEPTX_BEGIN_IDX:", conceptX_begin_idx)
					#print("CONCEPTX_END_IDX:", conceptX_end_idx)

				if conceptX_begin_idx == -1 or conceptX_end_idx == -1 or conceptY_begin_idx == -1 or conceptY_end_idx == -1:
					continue
				
				conceptX = question[conceptX_begin_idx:conceptX_end_idx].lower()
				conceptY = question[conceptY_begin_idx:conceptY_end_idx].lower()
				#print("conceptX:", conceptX)
				#print("conceptY:", conceptY)
			
			# Only X in the question:
			elif Ypos == -1:
				beforeX = q[:Xpos]
				afterX = q[Xpos+1:]
				
				concept_begin_idx = -1
				concept_end_idx = -1
				
				if question.find(beforeX) != -1:
					concept_begin_idx = Xpos # = len(beforeX)
				if question.find(afterX) != -1:
					concept_end_idx = len(question) - len(afterX)
				
				#print("ONLY_X")
				#print("CONCEPT_BEGIN_IDX:", concept_begin_idx)
				#print("CONCEPT_END_IDX:", concept_end_idx)

				if concept_begin_idx == -1 or concept_end_idx == -1:
					continue
				
				conceptX = question[concept_begin_idx:concept_end_idx].lower()
				#print("conceptX:", conceptX)

			# Only Y in the question:
			elif Xpos == -1:
				beforeY = q[:Ypos]
				afterY = q[Ypos+1:]
				
				concept_begin_idx = -1
				concept_end_idx = -1
				
				if question.find(beforeY) != -1:
					concept_begin_idx = Ypos
				if question.find(afterY) != -1:
					concept_end_idx = len(question) - len(afterY)
				
				#print("ONLY_Y")
				#print("CONCEPT_BEGIN_IDX:", concept_begin_idx)
				#print("CONCEPT_END_IDX:", concept_end_idx)
				
				if concept_begin_idx == -1 or concept_end_idx == -1:
					continue
					
				conceptY = question[concept_begin_idx:concept_end_idx].lower()
				#print("conceptY:", conceptY)
				
			for elem in self.knowledge_base:
				matchX = False
				matchY = False
			
				# X in the question:
				if Xpos != -1:
					c1 = elem["c1"]
					if c1.count("bn:") >= 2:
						pass
					elif "::" in c1:
						idx = c1.index("::")
						w = c1[:idx].lower()
						if conceptX in w:
							matchX = True
					elif "bn:" in c1:
						try:
							bn_conceptx = babelNetCache.cache[c1[c1.index("bn:"):]].lower()
							#print("bn_conceptx:", bn_conceptx)
							if conceptX in bn_conceptx or bn_conceptx in conceptX:
								matchX = True
						except:
							pass
					elif c1.lower() == conceptX:
						matchX = True

				# Y in the question:
				if Ypos != -1:
					c2 = elem["c2"]
					if c2.count("bn:") >= 2:
						pass
					elif "::" in c2:
						idx = c2.index("::")
						w = c2[:idx].lower()
						if conceptY in w:
							matchY = True
					elif "bn:" in c2:
						try:
							bn_concepty = babelNetCache.cache[c2[c2.index("bn:"):]].lower()
							#print("bn_concepty:", bn_concepty)
							if conceptY in bn_concepty or bn_concepty in conceptY:
								matchY = True
						except:
							pass
					elif c2.lower() == conceptY:
						matchY = True

				if Xpos != -1 and Ypos != -1:
					if matchX == True and matchY == True:
						#print("XY - Match found with:")
						#print(elem)
						return elem["answer"]
				elif matchX == True or matchY == True:
					#print("Match found with:")
					#print(elem)
					return elem["answer"]

		return "I don't understand."
