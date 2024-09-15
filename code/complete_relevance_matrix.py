import os
import csv
import yaml
import time
import json
import argparse
import itertools
import numpy as np
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
from multiprocessing import Pool, freeze_support


global_npy_dict = None
global_word2vec = None


def prepare_from_NPY(filepath_in: str):
    '''
    Retrieves data from a RELISH npy file, combining both title and abstract into a document.

    Parameters
    ----------
    filepath_in: str
            The filepath of the RELISH input npy file.
    Returns
    -------
    dict of nump array
            A dictionary where each tokenized document is stored at their pmid.
    '''
    doc = np.load(filepath_in, allow_pickle=True)
    dict = {}
    for line in doc:
        if isinstance(line[1], (np.ndarray, np.generic)):
            document = np.ndarray.tolist(line[1])
            document.extend(np.ndarray.tolist(line[2]))
        else:
            document = line[1]
            document.extend(line[2])
        dict[int(line[0])] = [w for w in document]
    return dict

def generate_param_combinations(params):
    param_keys = []
    param_values = []
    
    for key, value in params.items():
        if 'values' in value:  # Check if 'values' exist in this parameter
            param_keys.append(key)
            param_values.append(value['values'])
        else:
            param_keys.append(key)
            param_values.append([value['value']])  # Use the single value as a list
    
    param_combinations = [dict(zip(param_keys, combination)) 
                          for combination in itertools.product(*param_values)]
    
    return param_combinations


def generate_Word2Vec_model(params: dict, model_directory: str):
    '''
    Generates a word2vec model from all RELISH sentences using gensim and saves three model files.

    Parameters
    ----------
    filepath_in: list of str
            The filepath of the RELISH input npy file.
    params: dict
            A dictionary of the hyperparameters for the model.
    model_directory : str
            Directory to save the trained Word2vec model.
    '''
    from gensim.models import Word2Vec
    sentence_list = []
    for pmid in global_npy_dict:
        sentence_list.append(global_npy_dict[pmid])

    params['sentences'] = sentence_list
    model = Word2Vec(**params)
    model.save(model_directory)

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


def complete_relevance_matrix(evaluation_file: str, output_file: str):
    '''
    Adds Word Mover's Distance to the evaluation matrix from the .npy formatted embeddings.

    Parameters
    ----------
    evaluation_file: str
            The evaluation matrix csv file.
    output_file : str
            File path to relevance matrix with WMD values.
    '''
    start = time.time()

    print("Preparing rows...")
    header = []
    rows = []
    tokenset_pairs = []
    with open(evaluation_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        header = next(spamreader)  # Save and remove header
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
                # print(f"KeyError: {row[0]} or {row[1]} not found in dictionary")
                continue

    with open(output_file, 'w', newline='') as csvfile:
        header[-1] = 'WMD Similarity'
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
                    print(
                        f"Processed {total_processed}/{len(tokenset_pairs)} rows...")
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
    parser.add_argument("-p", "--params", type=str,
                        help="File location of word2vec parameter yaml file.")
    parser.add_argument("-o", "--output_directory", type=str,
                        help="Directory to store models and relevance matrices.")
    args = parser.parse_args()

    params = []
    with open(args.params, "r") as file:
        content = yaml.safe_load(file)
        params = content['params']

    param_combinations = generate_param_combinations(params)

    print("Preparing NPY Dict")
    global_npy_dict = prepare_from_NPY(args.input)

    for i, param_set in enumerate(param_combinations):
        print(f"Training model with hyperparameters: {param_set}")
        model_directory = f"{args.output_directory}/models"
        os.makedirs(model_directory, exist_ok=True)
        model_output_file = f"{model_directory}/word2vec_model_{i}.model"
        generate_Word2Vec_model(
            param_set, model_output_file)
        try:
            # Due to multiprocessing, this needs to be initialized here.
            global_word2vec = KeyedVectors.load(model_output_file)
        except:
            "Word embddings not definined yet, generating word embeddings..."
        freeze_support()
        matrix_directory = f"{args.output_directory}/matrix"
        os.makedirs(matrix_directory, exist_ok=True)
        matrix_output_file = f"{matrix_directory}/relevance_WMD_{i}.tsv"
        complete_relevance_matrix(args.matrix, matrix_output_file)