from pathlib import Path

# Cache babelnetid-lemma to reduce the queries to BabelNet (save babelcoins),
# speed up the training of concept_extractor.keras and the generation of questions:
class BabelNetCache:
	def __init__(self, cache_file_path):
		self.cache = {}
		self.cache_file_path = cache_file_path
		
		if Path(self.cache_file_path).is_file():
			cache_file = open(self.cache_file_path)
			lines = cache_file.readlines()
			cache_file_len = len(lines)
			print("Reading " + self.cache_file_path + " (" + str(cache_file_len) + " elements)")
			cnt = 0
			
			for elem in lines:
				cnt += 1
				print("Progress: {:2.1%}".format(cnt / cache_file_len), end="\r")
				elem = elem.split("\t")
				self.cache[elem[0]] = elem[1].rstrip("\n")

			print("\nDone.")
			cache_file.close()

	def save(self, cache_file_path=None):
		
		if cache_file_path is None:
			cache_file_path = self.cache_file_path
		
		cache_file = open(cache_file_path, "w")
		for babelnetid, lemma in self.cache.items():
			cache_file.write(babelnetid + "\t" + lemma + "\n")
		cache_file.close()
