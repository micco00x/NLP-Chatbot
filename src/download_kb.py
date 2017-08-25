import urllib.request
#from http.client import IncompleteRead
import json

BABELNET_KEY = "5aa541b8-e16d-4170-8e87-868c0bff9a5e"

SERVER_IP_ADDRESS = "151.100.179.26"
SERVER_PORT = 8080
SERVER_PATH = "/KnowledgeBaseServer/rest-api/"

knowledge_base = []

kb_size = int(urllib.request.urlopen("http://" + SERVER_IP_ADDRESS + ":" + str(SERVER_PORT) + SERVER_PATH +
									 "items_number_from?id=0&key=" + BABELNET_KEY).read())

print("kb_size:", kb_size)

cnt = 0
while cnt < kb_size:
	try:
		data = urllib.request.urlopen("http://" + SERVER_IP_ADDRESS + ":" + str(SERVER_PORT) + SERVER_PATH +
									  "items_from?id=" + str(cnt) + "&key=" + BABELNET_KEY).read().decode()
		knowledge_base.extend(json.loads(data))
	except: #IncompleteRead as e:
		# TODO: some data is lost here
		#print("ERROR (cnt: " + str(cnt) + ")")
		continue # make same request again, is there a better way to handle this?
		#print(str(e))

	print("Progress: {:2.1%}".format(cnt / kb_size), end="\r")
	cnt += 5000

print("Complete.")

kb_file = open("../resources/kb.json", "w")
kb_file.write(json.dumps(knowledge_base))
kb_file.close()
