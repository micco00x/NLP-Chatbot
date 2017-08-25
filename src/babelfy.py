import os

import nltk
from nltk.parse.stanford import StanfordDependencyParser
import itertools
from lxml import etree

import urllib.request, urllib.error, urllib.parse
import urllib.request, urllib.parse, urllib.error
import json
import gzip
from io import BytesIO

BABELNET_KEY  = "5aa541b8-e16d-4170-8e87-868c0bff9a5e"

#################### Relation Disambiguation (Babelfy) ####################

# Disambiguate a concept of a sentence using Babelfy,
# concept_start and concept_end are the token indices
# (concept_end is included):
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

		for result in data:
			tokenFragment = result.get("tokenFragment")
			tfStart = tokenFragment.get("start")
			tfEnd = tokenFragment.get("end")
			print((str(tfStart) + "\t" + str(tfEnd)))

			if tfStart == concept_start and tfEnd == concept_end:
				return result.get("babelSynsetID")

	return " ".join(sentence.split()[concept_start:concept_end+1])

print(babelfy_disambiguate("BabelNet is both a multilingual encyclopedic dictionary and a semantic network", 0, 1))
