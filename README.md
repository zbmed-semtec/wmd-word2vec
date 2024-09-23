[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

[![SWH](https://archive.softwareheritage.org/badge/swh:1:dir:db21ed332bddbccf48c9ee5856c8539096e74a64/)](https://archive.softwareheritage.org/swh:1:dir:db21ed332bddbccf48c9ee5856c8539096e74a64;origin=https://github.com/zbmed-semtec/wmd-word2vec;visit=swh:1:snp:03f94efb23a9d3a222a9087401081ad679b22f56;anchor=swh:1:rev:215c9bd3f9718c15a43d4163f6a1d46754097837)

# WMD-Word2Vec
This repository focuses on the RELISH Corpus to identify relevancy of a given pair of PubMed papers. The approach uses Word Mover's Distance (WMD), which computes a semantic closeness between two documents. WMD does not require any hyperparamters to function, however it is reliant on word embeddings, with which it can calculate the distance of word vectors between two documents.

## Table of Contents

1. [About](#about)
2. [Input Data](#input-data)
3. [Pipeline](#pipeline)
    1. [Generate Embeddings](#generate-embeddings)
        - [Using Trained Word2Vec models](#using-trained-word2vec-models)
          - [Parameters](#parameters)
          - [Hyperparameters](#hyperparameters)
        - [Using Pre-trained Word2Vec models](#using-pre-trained-word2vec-models)
    2. [Calculate Word Mover's Distance](#calculate-word-movers-distance-📐)
    3. [Evaluation](#evaluation)
        - [Precision@N](#precisionn)
        - [nDCG@N](#ndcgn)
4. [Code Implementation](#code-implementation)
5. [Getting Started](#getting-started)

## About

Our approach involves utilizing Word2Vec to capture word-level semantics and to generate word embeddings. We then employ the centroid approach to create document-level embeddings, which entails calculating the centroids of word embeddings within each document's title and abstract. For word embeddings, we utilize trained models. For our trained model, we utilize the documents from the RELISH dataset as the training corpus. The words from the titles and abstracts of each document are combined as the input to the model as a single document.

## Input Data

The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words. We also additionally remove stopwords from the tokenized input.

## Pipeline

The following section outlines the process of generating word-level embeddings for each PMID of the RELISH corpus and completing the similarity matrix with word mover's distances.

### Generate Embeddings
The following section species on the process of either training a word2vec model or using a pretrained one.

#### Using Trained Word2Vec models
We construct Word2Vec models with customizable hyperparameters. We employ the parameters shown below in order to generate our models.
##### Parameters

+ **sg:** {1,0} Refers to the training algorithm. If sg=1, skim grams is used otherwise, continuous bag of words (CBOW) is used.
+ **vector_size:** It represents the number of dimensions our embeddings will have.
+ **window:** It represents the maximum distance between the current and predicted word.
+ **epochs:** It is the nuber of iterations of the training dataset.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm.

#### Hyperparameters
The hyperparameters can be modified in [`hyperparameters.yaml`](./code/hyperparameters.yaml)

### Calculate Word Mover's Distance
To assess the similarity between two documents within the RELISH corpus, we use the Word Mover's Distance calculted via the gensim module for [Word Mover's Distance](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html). For this process we complete a 4-column relevance matrix with an adjusted WMD value to represent closeness between to documents.

## Evaluation

### Precision@N

In order to evaluate the effectiveness of this approach, we make use of Precision@N. Precision@N measures the precision of retrieved documents at various cutoff points (N).We generate a Precision@N matrix for existing pairs of documents within the RELISH corpus, based on the original RELISH JSON file. The code determines the number of true positives within the top N pairs and computes Precision@N scores. The result is a Precision@N matrix with values at different cutoff points, including average scores. For detailed insights into the algorithm, please refer to this [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Precision%40N_existing_pairs).

### nDCG@N

Another metric used is the nDCG@N (normalized Discounted Cumulative Gain). This ranking metric assesses document retrieval quality by considering both relevance and document ranking. It operates by using a TSV file containing relevance and word mover's closeness scores, involving the computation of DCG@N and iDCG@N scores. The result is an nDCG@N matrix for various cutoff values (N) and each PMID in the corpus, with detailed information available in the [documentation](https://github.com/zbmed-semtec/medline-preprocessing/tree/main/code/Evaluation).

## Code Implementation

The [`complete_relevance_matrix.py`](./code/complete_relevance_matrix.py) script uses the RELISH Tokenized npy file as input and supports the generation and training of Word2Vec models and completing the blank 4 column wide [relevance matrix](./data/relevance_WMD_blank.tsv) with the relevant Word Mover's Closeness scores for each pmid pair.

## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/wmd_word2vec.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/wmd_word2vec.git
```
### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Dataset

Use the Download_Dataset.sh script to download the Split Dataset by 
running the following commands:

```
chmod +777 Download_Dataset.sh
./Download_Dataset.sh
```

This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders as shown below.

```
📦 /wmd-word2vec
└─ data
   └─ Input
      ├─ Tokens
      │  ├─ relish.npy
      └─ Ground_truth
         └─ relevance_matrix.tsv
```


The file *relish.npy* is in the NumPy binary format (.npy), which is specifically used to store NumPy arrays efficiently. These arrays contain the PMID, title, and abstract for each document.

In contrast, *relevance_matrix.tsv* is a Tab-separated Values file, similar to CSV but using tabs as delimiters. It stores tabular data with four columns: PMID1 | PMID2 | Relevance | WMD Similarity.

Reference: Tab-separated values (TSV) file format:  
[![FAIRsharing DOI](https://img.shields.io/badge/DOI-10.25504%2FFAIRsharing.a978c9-blue)](https://doi.org/10.25504/FAIRsharing.a978c9)

### Step 4: Generate Embeddings and Calculate Word Mover's Distances
The [`complete_relevance_matrix.py`](./code/complete_relevance_matrix.py) script uses the RELISH Tokenized npy file as input and includes a default parameter json with preset hyperparameters. You can easily adapt it for different values and parameters by modifying the [`hyperparameters.yaml`](./code/hyperparameters.yaml). Make sure to have the RELISH Tokenized.npy file within the directory under the data folder.

```
python3 code/complete_relevance_matrix.py [-i RELISH FILE PATH] [-ma OUTPUT RELEANCE MATRIX PATH] [-p HYPERPARAMETERS YAML PATH] [-o OUTPUT DIRECTORY PATH]
```

To run this script, please execute the following command:

```
python3 code/complete_relevance_matrix.py -i data/Input/Tokens/relish.npy -ma data/Input/Ground_truth/relevance_matrix.tsv -p code/hyperparameters.yaml -o data/
```

The script will first create word embeddings and then compute the Word Mover's Distances, completing the given relevance matrix. If you are using the default hyperparameters, you can expect it to create 18 new relevance matrices for each trained word2vec model.


### Step 5: Precision@N
In order to calculate the Precision@N scores and execute this [script](/code/precision.py), run the following command:

```
python3 code/precision.py [-i WMD FILE PATH] [-o OUTPUT PATH] [-c CLASSES]
```

You must pass the following three arguments:

+ -i/ --wmd_file_path: path to the 4-column word mover's closeness existing pairs RELISH file: (tsv file)
+ -o/ --output_path: path to save the generated precision matrix: (tsv file)
+ -c/ --classes: Number of classes for class distribution (2 or 3)

For example, if you are running the code from the code folder and have the word mover's closeness TSV file in the data folder, run the precision matrix creation for the first hyperparameter as:

```
python3 code/precision.py -i data/matrix/relevance_WMD_0.tsv -o data/precision_three_classes/precision_WMD_0.tsv -c 3
```
Note: You would have to run the above command for every hyperparameter configuration by changing the file name for the cosine similarity file or use the following shell script to generate all files at once.


```
for VALUE in {0..17};do
python3 code/precision.py -i data/matrix/relevance_WMD_${VALUE}.tsv -o data/precision_three_classes/precision_WMD_${VALUE}.tsv -c 3
done
```


### Step 6: nDCG@N
In order to calculate nDCG scores and execute this [script](/code/calculate_gain.py), run the following command:

```
python3 code/calculate_gain.py [-i INPUT]  [-o OUTPUT]
```

You must pass the following two arguments:

+ -i / --input: Path to the 4 column WMD similarity existing pairs RELISH TSV file.
+ -o/ --output: Output path along with the name of the file to save the generated nDCG@N TSV file.

For example, if you are running the code from the code folder and have the 4 column RELISH TSV file in the data folder, run the matrix creation for the first hyperparameter as:

```
python3 code/calculate_gain.py -i data/matrix/relevance_WMD_0.tsv -o data/gain_matrices/nCDG_WMD_0.tsv
```
Note: You would have to run the above command for every hyperparameter configuration by changing the file name for the cosine similarity file or use the following shell script to generate all files at once.

```
for VALUE in {0..17};do
python3 code/calculate_gain.py -i data/matrix/relevance_WMD_${VALUE}.tsv -o data/gain_matrices/ndcg_${VALUE}.tsv
done
```


### Step 7: Compile Results

In order to compile the average result values for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.


If you are running the code from the code folder, run the compilation script as:

```
python3 code/evaluation/show_avg.py -i data/gain_matrices/ -o data/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path.