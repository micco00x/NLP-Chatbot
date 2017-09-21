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
import torch

import numpy as np
import re

from AnswerGenerator import AnswerGenerator
from BabelNetCache import BabelNetCache
from QuestionGenerator import QuestionGenerator
from QuestionPatterns import QuestionPatterns
from utils import *
from Word2Vec import *

from seq2seq.Seq2Seq import Seq2Seq
import seq2seq.utils

import sys

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

# Question patterns:
questionPatterns = QuestionPatterns("../resources/patterns.tsv")

# Question generator:
question_generator = QuestionGenerator("../babelnet/BabelDomains_full/BabelDomains/babeldomains_babelnet.txt",
									   "../resources/domains_to_relations.tsv",
									   questionPatterns)
									   #"../resources/patterns.tsv")

# BabelNetCache:
babelNetCache = BabelNetCache("../resources/babelnet_cache.tsv")

# HParams from json file (command line args):
with open(sys.argv[1]) as hparams_file:
	hparams = json.load(hparams_file)

# HParams for answer generator:
hparams_answer_generator = hparams["answerGenerator"]
hparams_relation_classifier = hparams["relationClassifier"]
hparams_concept_extractor = hparams["conceptExtractor"]

# Vocabularies for seq2seq encoder/decoder:
vocabulary_encoder = Vocabulary(hparams_answer_generator["encoderVocabularyPath"])
vocabulary_decoder = Vocabulary(hparams_answer_generator["decoderVocabularyPath"])
relation_classifier_vocabulary = Vocabulary(hparams_relation_classifier["vocabularyPath"])
concept_extractor_vocabulary = Vocabulary(hparams_concept_extractor["vocabularyPath"])

# Load NN models:
print("Loading NN models...")
relation_classifier = keras.models.load_model("../models/relation_classifier.keras")
concept_extractor = keras.models.load_model("../models/concept_extractor.keras")
graph = tf.get_default_graph()

seq2seq_model = Seq2Seq("eval",
						vocabulary_encoder.VOCABULARY_DIM, vocabulary_decoder.VOCABULARY_DIM,
						vocabulary_decoder.word2index[vocabulary_decoder.PAD_SYMBOL],
						vocabulary_decoder.word2index[vocabulary_decoder.GO_SYMBOL],
						vocabulary_decoder.word2index[vocabulary_decoder.EOS_SYMBOL],
						hparams_answer_generator["encoderHiddenSize"],
						hparams_answer_generator["decoderHiddenSize"],
						300, # TODO: this should be included in hparams too
						embedding_padding_idx=vocabulary_decoder.word2index[vocabulary_decoder.PAD_SYMBOL])
# TODO: loading on GPU with CUDA shouldn't create any problem (i.e. you should not need to remove some items)
state_dict = torch.load(hparams_answer_generator["checkpoint"], map_location=lambda storage, loc:storage)["state_dict"]
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
	if not k.endswith("l1"):
		new_state_dict[k] = v
seq2seq_model.load_state_dict(new_state_dict)
seq2seq_model = seq2seq_model.cuda() if torch.cuda.is_available() else seq2seq_model

print("Done.")

# Open the Knowledge Base:
print("Loading the knowledge base...")
with open("../resources/kb.json") as kb_file:
	knowledge_base = json.load(kb_file)
print("Done.")

# Answer generator:
answerGenerator = AnswerGenerator(knowledge_base, questionPatterns,
								  relation_classifier, relation_classifier_vocabulary)

# Dictionary of user status (manage multiple users):
user_status = {}

# BabelNet domains:
babelnet_domains = []
babelnet_domain_list_file = open("../babelnet/BabelDomains_full/domain_list.txt")
for line in babelnet_domain_list_file:
	babelnet_domains.append(line.rstrip())

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
			if random.uniform(0, 1) < 1.0: # 0.5
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
			print("answerGenerator:", answerGenerator.generate(msg["text"], babelNetCache, graph))
			q_qaNN = vocabulary_encoder.sentence2indices(user_status[chat_id].question)
			q_qaNN.reverse() # Q&A NN uses reversed sentence
			q_rcNN = relation_classifier_vocabulary.sentence2indices(user_status[chat_id].question)
			with graph.as_default():
				user_status[chat_id].relation = int_to_relation(np.argmax(relation_classifier.predict(np.array(q_rcNN))[0]))
			# TODO: send correct answer using the model
			encoder_input = torch.autograd.Variable(torch.LongTensor([q_qaNN]))
			decoder_input = torch.autograd.Variable(torch.LongTensor([[seq2seq_model.GO_SYMBOL_IDX]]))
			if torch.cuda.is_available():
				encoder_input = encoder_input.cuda()
				decoder_input = decoder_input.cuda()
			answer_idx = seq2seq_model(encoder_input, decoder_input, 30)
			answer = " ".join([vocabulary_decoder.index2word[w_idx] for w_idx in answer_idx[:-1]])
			bot.sendMessage(chat_id, answer + " The relation is " + user_status[chat_id].relation)
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
					c2_probability_concept = concept_extractor.predict(np.array(concept_extractor_vocabulary.sentence2indices(answer)))
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
				c2 = user_status[chat_id].question_data["id2"]
				with graph.as_default():
					c1_probability_concept = concept_extractor.predict(np.array(concept_extractor_vocabulary.sentence2indices(answer)))
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
				data_c1 = c1
				data_c2 = user_status[chat_id].question_data["c2"] + "::" + c2
			data = {
				"question": user_status[chat_id].question,
				"answer": answer,
				"relation": user_status[chat_id].relation,
				"context": "C",
				"domains": [user_status[chat_id].domain],
				"c1": data_c1,
				"c2": data_c2
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
