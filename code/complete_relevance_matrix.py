import argparse
from gensim.models import KeyedVectors
from multiprocessing import Pool, freeze_support
import csv
from nltk.corpus import stopwords
import numpy as np
from gensim.models import KeyedVectors
import time
global_npy_dict = None
global_word2vec = None
try:
        global_word2vec = KeyedVectors.load("./data/word2vec_model") # Due to multiprocessing, this needs to be initialized here.
except:
        "Word embddings not definined yet, generating word embeddings..."

def prepareFromNPY(filepathIn: str):
        '''
        Retrieves data from a RELISH npy file, further removing any stopword tokens and combining both title and abstract into a document.

        Parameters
        ----------
        filepathIn: str
                The filepath of the RELISH input npy file.

        Returns
        -------
        dict of nump array
                A dictionary where each tokenized document is stored at their pmid.
        '''
        stop_words = set(stopwords.words('english'))
        doc = np.load(filepathIn, allow_pickle=True)
        dict = {}
        for line in doc:
                document = np.ndarray.tolist(line[1])
                document.extend(np.ndarray.tolist(line[2]))
                dict[np.ndarray.tolist(line[0])] = [w for w in document if not w in stop_words]
        return dict

def generateWord2VecModel(filepathIn: str, params: dict):
        '''
        Generates a word2vec model from all RELISH sentences using gensim and saves three model files.

        Parameters
        ----------
        filepathIn: list of str
                The filepath of the RELISH input npy file.
        params: dict
                A dictionary of the hyperparameters for the model.
        '''
        from gensim.models import Word2Vec
        dictionary = prepareFromNPY(filepathIn)
        sentenceList = []
        for pmid in dictionary:
                sentenceList.append(dictionary[pmid])
        params['sentences'] = sentenceList
        model = Word2Vec(**params)
        model.save("./data/word2vec_model")
                

def getWMDDistance(tokens: list):
        '''
        Computes the word mover's distance between two documents.
        Used in completeRelevanceMatrix with multiprocessing.

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


def completeRelevanceMatrix(EvaluationFile: str):
        '''
        Adds Word Mover's Distance to the evaluation matrix from the .npy formatted embeddings.

        Parameters
        ----------
        EvaluationFile: str
                The evaluation matrix csv file.
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
                                rows.append(row)
                                tokenset_pairs.append((
                                        global_npy_dict[row[0]],
                                        global_npy_dict[row[1]]
                                ))
                        except KeyError:
                                print(f"KeyError: {row[0]} or {row[1]} not found in dictionary")
                                continue

        print(f"Processing {len(tokenset_pairs)} rows...")

        with open(EvaluationFile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow(header)

                total_processed = 0
                with Pool() as p:
                        iterator = p.imap(getWMDDistance, tokenset_pairs, 100)
                        for distance in iterator:
                                row = rows[total_processed]
                                row[3] = round(1/(1+distance), 2)
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
        parser.add_argument("-m", "--matrix", type=str,
                       help="Path of relevance matrix file")                
        args = parser.parse_args()

        params = {'vector_size':200, 'epochs':5, 'window':5, 'min_count':2, 'workers':4}

        print("Preparing NPY dict...")
        global_npy_dict = prepareFromNPY(args.input)
        generateWord2VecModel(args.input, "./data/word2vec_model", params)
        global_word2vec = KeyedVectors.load("./data/word2vec_model") 
        freeze_support()
        completeRelevanceMatrix(args.matrix)