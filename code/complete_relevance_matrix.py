import argparse
from gensim.models import KeyedVectors
from multiprocessing import Pool, freeze_support
import csv
import numpy as np
from gensim.models import KeyedVectors
import time
import json
import os
global_npy_dict = None
global_word2vec = None
try:
	global_word2vec = KeyedVectors.load("./data/word2vec_model") # Due to multiprocessing, this needs to be initialized here.
except:
	"Word embddings not definined yet, generating word embeddings..."

def prepare_from_NPY(filepath_in: str, remove_stop_words: bool):
	'''
	Retrieves data from a RELISH npy file, further removing any stopword tokens and combining both title and abstract into a document.

	Parameters
	----------
	filepath_in: str
		The filepath of the RELISH input npy file.
	remove_stop_words: bool
		Whether stopwords get removed or not.
	Returns
	-------
	dict of nump array
		A dictionary where each tokenized document is stored at their pmid.
	'''
	doc = np.load(filepath_in, allow_pickle=True)
	dict = {}
	if remove_stop_words:
		import nltk
		nltk.download('stopwords')
		from nltk.corpus import stopwords
		stop_words = set(stopwords.words('english'))
		for line in doc:
			document = np.ndarray.tolist(line[1])
			document.extend(np.ndarray.tolist(line[2]))
			print(line[0])
			dict[int(line[0])] = [w for w in document if not w in stop_words]
	else:
		for line in doc:
			document = np.ndarray.tolist(line[1])
			document.extend(np.ndarray.tolist(line[2]))
			dict[int(line[0])] = [w for w in document]
	return dict

def generate_Word2Vec_model(params: dict, iteration: int, word_embedding_directory: str):
	'''
	Generates a word2vec model from all RELISH sentences using gensim and saves three model files.

	Parameters
	----------
	filepath_in: list of str
		The filepath of the RELISH input npy file.
	params: dict
		A dictionary of the hyperparameters for the model.
	iteration: int
		The number of hyperparameters processed.
	'''
	from gensim.models import Word2Vec
	sentence_list = []
	for pmid in global_npy_dict:
		sentence_list.append(global_npy_dict[pmid])
	
	params['sentences'] = sentence_list
	model = Word2Vec(**params)
	model.save("./data/word2vec_model") # This has to be static
	os.makedirs(f"{word_embedding_directory}/{iteration}", exist_ok=True)
	model.save(f"{word_embedding_directory}/{iteration}/word2vec_model")

def get_WMD_distance(tokens: list):
	'''
	Computes the word mover's distance between two documents.
	Used in complete_relevance_matrix with multiprocessing.

	Parameters
	----------
	tokens: list of strings
		Tokenized document pair
	Returns
	-------
	float
		The rounded Word Mover's Distance (WMD).
	'''
	return global_word2vec.wv.wmdistance(tokens[0], tokens[1])


def complete_relevance_matrix(evaluation_file: str, iteration: int):
	'''
	Adds Word Mover's Distance to the evaluation matrix from the .npy formatted embeddings.

	Parameters
	----------
	evaluation_file: str
		The evaluation matrix csv file.
	iteration: int
		The number of hyperparameters processed.
	'''
	start = time.time()

	print("Preparing rows...")
	header = []
	rows = []
	tokenset_pairs = []
	with open("./data/relevance_WMD_blank.tsv", newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter='\t')
		header = next(spamreader) # Save and remove header
		for row in spamreader:
			try: 
				first_doc = global_npy_dict[int(row[0])]
				second_doc = global_npy_dict[int(row[1])]
				rows.append(row)
				tokenset_pairs.append((
					first_doc,
					second_doc
				))
			except KeyError:
				#print(f"KeyError: {row[0]} or {row[1]} not found in dictionary")
				continue
	print(f"Processing {len(tokenset_pairs)} rows...")

	with open(f"{evaluation_file}_{iteration}.tsv", 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter='\t')
		writer.writerow(header)

		total_processed = 0
		with Pool() as p:
			iterator = p.imap(get_WMD_distance, tokenset_pairs, 100)
			for distance in iterator:
				row = rows[total_processed]
				row[3] = round(1/(1+distance), 4)
				writer.writerow(row)

				total_processed += 1
				if total_processed % 100 == 0 or total_processed == len(tokenset_pairs):
					print(f"Processed {total_processed}/{len(tokenset_pairs)} rows...")
		p.join()
		p.close()
		global_word2vec = None
	print(time.time() - start)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str,
		help="Path to input RELISH tokenized .npy file")
	parser.add_argument("-ma", "--matrix", type=str,
		help="Path of relevance matrix file")
	parser.add_argument("-s", "--rm_stopwords", type=int,
		help="Whether to remove stopwords or not")
	parser.add_argument("-pj", "--params_json", type=str,
		help="File location of word2vec parameter list.")
	parser.add_argument("-md", "--model_directory", type=str,
		help="Directory to store word embeddings in.")   
	args = parser.parse_args()


	params = []
	with open(args.params_json, "r") as openfile:
		params = json.load(openfile)

	print("Preparing NPY dict...")
	global_npy_dict = prepare_from_NPY(args.input, args.rm_stopwords)
	for iteration in range(len(params)):
		generate_Word2Vec_model(params[iteration], iteration, args.model_directory)
		global_word2vec = KeyedVectors.load("./data/word2vec_model") 
		freeze_support()
		complete_relevance_matrix(args.matrix, iteration)