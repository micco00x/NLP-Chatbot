from enum import Enum

import http.client
import json

import random

import telepot
from pprint import pprint

from telepot.loop import MessageLoop

import time

import keras
import tensorflow as tf

import numpy as np
import re

from QuestionGenerator import *
from utils import *
from Word2Vec import *


CHATBOT_TOKEN = "436863628:AAFHt0_YkqbjyMnoIBOltdntdRFYxd4Q-XQ"
BABELNET_KEY = "5aa541b8-e16d-4170-8e87-868c0bff9a5e"

SERVER_IP_ADDRESS = "151.100.179.26"
SERVER_PORT = 8080
SERVER_PATH = "/KnowledgeBaseServer/rest-api/"

# User status (for each user the bot has a different behaviour):
class USER_STATUS(Enum):
	STARTING_CONVERSATION = 0
	CHOOSING_DOMAIN = 1
	ASKING_QUESTION = 2
	ASKING_RELATION = 3
	ANSWERING_QUESTION = 4

class User_Status:
	def __init__(self):
		self.status = USER_STATUS.STARTING_CONVERSATION
		self.domain = ""
		self.question = ""
		self.question_data = ""
		self.relation = ""

# Question generator:
question_generator = QuestionGenerator("../babelnet/BabelDomains_full/BabelDomains/babeldomains_babelnet.txt",
									   "../resources/domains_to_relations.tsv",
									   "../resources/patterns.tsv")

# BabelNetCache:
babelNetCache = BabelNetCache("../resources/babelnet_cache.tsv")

# Word2Vec:
print("Loading Word2Vec...")
word2vec = Word2Vec("../resources/Word2Vec.bin", "../resources/vocabulary.txt")
print("Done.")

# Load NN models:
print("Loading NN models...")
relation_classifier = keras.models.load_model("../models/relation_classifier.keras")
concept_extractor = keras.models.load_model("../models/concept_extractor.keras")
print("Done.")
graph = tf.get_default_graph()

# Dictionary of user status (manage multiple users):
user_status = {}

# BabelNet domains:
babelnet_domains = []
babelnet_domain_list_file = open("../babelnet/BabelDomains_full/domain_list.txt")
for line in babelnet_domain_list_file:
	babelnet_domains.append(line.rstrip())

# Setting up connection with server:
conn = http.client.HTTPConnection(SERVER_IP_ADDRESS, SERVER_PORT)
conn.request("GET", SERVER_PATH + "items_number_from?id=0&key=" + BABELNET_KEY)
r1 = conn.getresponse()
conn.close()
data1 = r1.read()

# Setting up Telegram bot:
bot = telepot.Bot(CHATBOT_TOKEN)
#print(bot.getMe())

response = bot.getUpdates()
#pprint(response)

# Handle message:
def handle(msg):
	global graph
	
	content_type, chat_type, chat_id = telepot.glance(msg)
	#print(content_type, chat_type, chat_id)
			
	if content_type == "text":
		print(str(chat_id) + ":", msg["text"])
		
		if chat_id not in user_status:
			bot.sendMessage(chat_id, "Hi!")
			user_status[chat_id] = User_Status()
	
		if user_status[chat_id].status == USER_STATUS.CHOOSING_DOMAIN:
			# TODO: recognize domain via NN (you need a dataset)
			recognized_domain = recognize_domain(babelnet_domains, msg["text"])
			print("Recognizing domain:", msg["text"], "->", recognized_domain)
			user_status[chat_id].domain = recognized_domain
			if random.uniform(0, 1) < 0.0: # 0.5
				bot.sendMessage(chat_id, "Ask me anything when you're ready then!")
				user_status[chat_id].status = USER_STATUS.ASKING_QUESTION
			else:
				question_chosen = False
				while not question_chosen:
					try:
						question_data = question_generator.generate(user_status[chat_id].domain, babelNetCache)
						question_chosen = True
					except:
						pass

				user_status[chat_id].question_data = question_data
				user_status[chat_id].question = question_data["question"]
				user_status[chat_id].relation = question_data["relation"]
				bot.sendMessage(chat_id, user_status[chat_id].question)
				user_status[chat_id].status = USER_STATUS.ANSWERING_QUESTION
		elif user_status[chat_id].status == USER_STATUS.ASKING_QUESTION:
			user_status[chat_id].question = msg["text"]
			q = word2vec.s2i(user_status[chat_id].question)
			with graph.as_default():
				user_status[chat_id].relation = int_to_relation(np.argmax(relation_classifier.predict(np.array(q))[0]))
			# TODO: send correct answer using the model
			bot.sendMessage(chat_id, "Here's your answer! The relation is " + user_status[chat_id].relation)
			user_status[chat_id].status = USER_STATUS.STARTING_CONVERSATION
		elif user_status[chat_id].status == USER_STATUS.ANSWERING_QUESTION:
			answer = msg["text"]
			answer_split = answer.split()
			answer_punctuation_split = split_words_punctuation(answer)
			bot.sendMessage(chat_id, "Alright! Thanks!")
			
			if user_status[chat_id].question_data["type"] == "XY":
				c1 = user_status[chat_id].question_data["id1"]
				c2 = user_status[chat_id].question_data["id2"]
				data_c1 = user_status[chat_id].question_data["c1"] + "::" + c1
				data_c2 = user_status[chat_id].question_data["c2"] + "::" + c2
			elif user_status[chat_id].question_data["type"] == "X":
				c1 = user_status[chat_id].question_data["id1"]
				with graph.as_default():
					c2_probability_concept = concept_extractor.predict(np.array(word2vec.s2i(answer)))
				print(c2_probability_concept)
				c2_tokens = probabilities_to_concept_tokens(c2_probability_concept)
				print("c2_tokens:", c2_tokens)
				# NN tokens indices to BabelNet token indices:
				for idx, w in enumerate(answer_split):
					if answer_punctuation_split[c2_tokens[0]] in w:
						c2_tokens[0] = idx
						break
				for idx, w in enumerate(answer_split):
					if answer_punctuation_split[c2_tokens[1]] in w:
						c2_tokens[1] = idx
				print("c2_tokens:", c2_tokens)
				c2 = babelfy_disambiguate(answer, c2_tokens[0], c2_tokens[1])
				data_c1 = user_status[chat_id].question_data["c1"] + "::" + c1
				data_c2 = c2
			else:
				with graph.as_default():
					c1_probability_concept = concept_extractor.predict(np.array(word2vec.s2i(answer)))
				print(c1_probability_concept)
				c1_tokens = probabilities_to_concept_tokens(c1_probability_concept)
				print("c1_tokens:", c1_tokens)
				# NN tokens indices to BabelNet token indices:
				for idx, w in enumerate(answer_split):
					if answer_punctuation_split[c1_tokens[0]] in w:
						c1_tokens[0] = idx
						break
				for idx, w in enumerate(answer_split):
					if answer_punctuation_split[c1_tokens[1]] in w:
						c1_tokens[1] = idx
				print("c1_tokens:", c1_tokens)
				c1 = babelfy_disambiguate(answer, c1_tokens[0], c1_tokens[1])
				w1 = "" # word already in c1 (format w::bn:--n)
				data_c2 = user_status[chat_id].question_data["id2"] + "::" + c2
			data = {
				"question": user_status[chat_id].question,
				"answer": answer,
				"relation": user_status[chat_id].relation,
				"context": "C",
				"domains": [user_status[chat_id].domain],
				"c1": data_c1, # TO TEST: w::bn:--n"
				"c2": data_c2  # TO TEST: w::bn:--n"
			}

			print("Enriching KB with:")
			print(data)

			conn = http.client.HTTPConnection(SERVER_IP_ADDRESS, SERVER_PORT)
			conn.request("POST",
						 SERVER_PATH + "add_item_test?key=" + BABELNET_KEY,
						 json.dumps(data),
						 {"Content-type": "application/json"})

			# TODO: manage answer from server (1/-1) and errors
			r1 = conn.getresponse()
			conn.close()
			print(r1.status, r1.reason)
			data1 = r1.read()
			print(data1)

			user_status[chat_id].status = USER_STATUS.STARTING_CONVERSATION

		if user_status[chat_id].status == USER_STATUS.STARTING_CONVERSATION:
			bot.sendMessage(chat_id, "What would you like to talk about?")
			user_status[chat_id].status = USER_STATUS.CHOOSING_DOMAIN

MessageLoop(bot, handle).run_as_thread()
print("The bot is ready.")

# Keep the program running:
while 1:
	time.sleep(10)

# TODO: create a thread to take commands from command line
#		e.g. close the bot
# TODO: # Save the cache with the new elements found from the queries:
#		babelNetCache.save()
