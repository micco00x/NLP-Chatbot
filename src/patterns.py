# Takes all the pattern files and creates a single pattern file
# with all the patterns, this should be then modified by hand
# in order to avoid duplications and poorly made question
# answer pairs (remember to remove bad files before running
# the program).

import os
from operator import itemgetter

patterns = []

for (dirpath, dirnames, filenames) in os.walk("../patterns"):
	for pattern_file in filenames:
		with open(dirpath + "/" + pattern_file, encoding="ISO-8859-1") as pf:
			for l in pf:
				elem = l.replace('"', "").rstrip("\n").lower().split("\t")
				if len(elem) == 2:
					elem[0] = elem[0].strip().rstrip().capitalize()
					
					# Makes "x" capital (it works with most of the cases), double check it by hand:
					try:
						xpos = elem[0].index("x")
						elem[0] = elem[0][:xpos] + elem[0][xpos:].capitalize()
					except:
						pass
					
					# Makes "y" capital (it works with most of the cases), double check it by hand:
					try:
						ypos = elem[0].index("y")
						elem[0] = elem[0][:ypos] + elem[0][ypos:].capitalize()
					except:
						pass
					
					elem[1] = elem[1].strip().rstrip()
					patterns.append(elem)

patterns = sorted(patterns, key=itemgetter(1))

pattern_file = open("../resources/patterns.tsv", "w")
for elem in patterns:
	pattern_file.write(elem[0] + "\t" + elem[1] + "\n")
pattern_file.close()
