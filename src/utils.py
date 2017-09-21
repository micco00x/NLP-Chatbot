import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
import json
import gzip
from io import BytesIO

import re

import numpy as np

BABELNET_KEY  = "5aa541b8-e16d-4170-8e87-868c0bff9a5e"

# Query to BabelNet to get the lemma of the BabelNetID:
def babelNetIdToLemma(babelNetID, babelNetCache=None):
	
	# Check if babelNetID is already in the cache:
	if babelNetCache is not None:
		try:
			lemma = babelNetCache.cache[babelNetID]
			#print(babelNetID + " is in cache")
			return lemma
		except:
			pass

	params = {
		"id" : babelNetID,
		"key"  : BABELNET_KEY
	}
	
	url = "https://babelnet.io/v4/getSynset?" + urllib.parse.urlencode(params)
	request = urllib.request.Request(url)
	request.add_header("Accept-encoding", "gzip")
	response = urllib.request.urlopen(request)

	if response.info().get("Content-Encoding") == "gzip":
		buf = BytesIO(response.read())
		f = gzip.GzipFile(fileobj=buf)
		data = json.loads(f.read().decode("utf-8"))
		w = str(data["senses"][0].get("lemma")).replace("_", " ")
	
	# Update the cache:
	if babelNetCache is not None:
		babelNetCache.cache[babelNetID] = w
	
	return w

# Disambiguate a concept of a sentence using Babelfy,
# concept_start and concept_end are the token indices
# (concept_end is included), returns the babelNetID
# if the disambiguation is available, the part of the
# sentence containing the concept otherwise:
def babelfy_disambiguate(sentence, concept_start, concept_end):
	params = {
		"text" : sentence,
		"lang" : "EN",
		"key"  : BABELNET_KEY
	}

	url = "https://babelfy.io/v1/disambiguate?" + urllib.parse.urlencode(params)
	request = urllib.request.Request(url)
	request.add_header("Accept-encoding", "gzip")
	response = urllib.request.urlopen(request)

	if response.info().get("Content-Encoding") == "gzip":
		buf = BytesIO(response.read())
		f = gzip.GzipFile(fileobj=buf)
		data = json.loads(f.read().decode("utf-8"))
		
		#print("Babelfy disambiguation results:")
		for result in data:
			tokenFragment = result.get("tokenFragment")
			tfStart = tokenFragment.get("start")
			tfEnd = tokenFragment.get("end")
			#print(("\t" + str(tfStart) + "\t" + str(tfEnd)))
			
			if tfStart == concept_start and tfEnd == concept_end:
				return " ".join(sentence.split()[concept_start:concept_end+1]) + "::" + result.get("babelSynsetID")

	return " ".join(sentence.split()[concept_start:concept_end+1])

# Given a set of N pairs of probabilities representing
# (Begin+End, Begin (but not End), End (but not Begin), Other (not Begin nor End)),
# (see concept extractor implementation in train.py), returns the first and the
# last indices (of NN) corresponding to concept tokens:
def probabilities_to_concept_tokens(probabilities):
	# Search for highest P between Begin+End and Begin token:
	B = 0   # index of begin token
	B_p = 0 # probability at begin token index
	is_begin = False
	for x in range(len(probabilities)):
		argmax = np.argmax(probabilities[x][0])
		if probabilities[x][0][argmax] > B_p and argmax < 2:
			B = x
			B_p = probabilities[x][0][argmax]
			if argmax == 0:
				is_begin = False
			if argmax == 1:
				is_begin = True

	if is_begin == False:
		return [B, B]

	E = B   # index of end token
	E_p = 0 # probability at end token index
	for x in range(B, len(probabilities)):
		argmax = np.argmax(probabilities[x][0])
		if probabilities[x][0][argmax] > E_p and argmax == 2:
			E = x
			E_p = probabilities[x][0][argmax]

	return [B, E]

# Convert relation string to int:
def relation_to_int(relation):
	return {"ACTIVITY": 0, "COLOR": 1, "GENERALIZATION": 2, "HOW_TO_USE": 3, "MATERIAL": 4, "PART": 5, "PLACE": 6,
			"PURPOSE": 7, "SHAPE": 8, "SIMILARITY": 9, "SIZE": 10, "SMELL": 11, "SOUND": 12, "SPECIALIZATION": 13,
			"TASTE": 14, "TIME": 15}[relation]

# Convert int to relation string:
def int_to_relation(integer):
	return ["ACTIVITY", "COLOR", "GENERALIZATION", "HOW_TO_USE", "MATERIAL", "PART", "PLACE", "PURPOSE", "SHAPE",
			"SIMILARITY", "SIZE", "SMELL", "SOUND", "SPECIALIZATION", "TASTE", "TIME"][integer]

# Split a sentence into words and punctuation symbols:
def split_words_punctuation(sentence):
	return re.findall(r"[\w]+|[^\s\w]", sentence)

# Find a pattern p in s (O(|s|*|p|)),
# since p is almost always small on avg. the
# algorithm runs in O(|s|):
def find_pattern(s, p):
	for i in range(len(s)-len(p)+1):
		match = True
		for j in range(len(p)):
			if s[i+j] != p[j]:
				match = False
				break
		if match == True:
			return i
	return -1

# Recognize the correct domain given a list of domains using Levenshtein distance:
def recognize_domain(domain_list, domain):
	domain_list_lower = [elem.lower() for elem in domain_list]
	domain_lower = domain.lower()

	dmin_index = 0
	dmin_v = levenshtein(domain_list_lower[0], domain_lower)

	for i in range(1, len(domain_list_lower)):
		d = levenshtein(domain_list_lower[i], domain_lower)
		if d < dmin_v:
			dmin_index = i
			dmin_v = d

	return domain_list[dmin_index]

# Christopher P. Matthews
# christophermatthews1985@gmail.com
# Sacramento, CA, USA
# From: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s, t):
	''' From Wikipedia article; Iterative with two matrix rows. '''
	if s == t: return 0
	elif len(s) == 0: return len(t)
	elif len(t) == 0: return len(s)
	v0 = [None] * (len(t) + 1)
	v1 = [None] * (len(t) + 1)
	for i in range(len(v0)):
		v0[i] = i
	for i in range(len(s)):
		v1[0] = i + 1
		for j in range(len(t)):
			cost = 0 if s[i] == t[j] else 1
			v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
		for j in range(len(v0)):
			v0[j] = v1[j]
	
	return v1[len(t)]
