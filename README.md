# WMD-word2vec
This repository focuses on the RELISH Corpus to identify relevancy of a given pair of PubMed papers.

# Approach
This approach uses Word Mover's Distance (WMD), which computes a semantic closeness between two documents. WMD does not require any hyperparamters to function, however it is reliant on word embeddings, with which it can calculate the distance of word vectors between two documents.

Our approach involves utilizing Word2Vec to capture word-level semantics and to generate word embeddings. We then employ the centroid approach to create document-level embeddings, which entails calculating the centroids of word embeddings within each document's title and abstract. For word embeddings, we utilize both pretrained and trained models.

By default, we use the "word2vec-google-news-300" Gensim word2vec model as the pretrained model for word embeddings. However, it can be substituted with other pretrained models available in the Gensim library.

For our trained model, we utilize the documents from the RELISH dataset as the training corpus. The words from the titles and abstracts of each document are combined and input into the model as a single document.

# Input Data
The input data for this method consists of preprocessed tokens derived from the RELISH documents. These tokens are stored in the RELISH.npy file, which contains preprocessed arrays comprising PMIDs, document titles, and abstracts. These arrays are generated through an extensive preprocessing pipeline, as elaborated in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing). Within this preprocessing pipeline, both the title and abstract texts undergo several stages of refinement: structural words are eliminated, text is converted to lowercase, and finally, tokenization is employed, resulting in arrays of individual words. We also additionally remove stopwords from the tokenized input.

# Generating Embeddings
The following section outlines the process of generating wordembeddings for each PMID of the RELISH corpus.

## Word embeddings

### Utilizing Trained Word2Vec models
We construct Word2Vec models with customizable hyperparameters. We employ the parameters shown below in order to generate our models.
#### Parameters

+ **dm:** {1,0} Refers to the training algorithm. If dm=1, distributed memory is used otherwise, distributed bag of words is used.
+ **vector_size:** It represents the number of dimensions our embeddings will have.
+ **window:** It represents the maximum distance between the current and predicted word.
+ **epochs:** It is the nuber of iterations of the training dataset.
+ **min_count:** It is the minimum number of appearances a word must have to not be ignored by the algorithm.

# Hyperparameter Optimization
*To be written*

# Code Implementation
The `complete_relevance_matrix.py` script uses the RELISH Tokenized npy file as input and supports the generation and training of Word2Vec models, generation of embeddings and saving the embeddings. After the embeddings have been created, the template blank relevance matrix will be completed using the calculated word mover's distance and normalizing it to a closeness value. The script includes a default parameter dictionary with present hyperparameters. You can easily adapt it for different values and parameters by modifying the `params` dictionary. With the

The script consists of two main functions `generate_Word2Vec_model` and `complete_relevance_matrix`.

`generate_Word2Vec_model` : This function creates a Word2vec model using the provided sentences and the inputer hyper parameter.

`complete_relevance_matrix` :  This function completes the template relevance matrix by adding calculating and adding the word mover's closeness for each pair. The word mover's closeness is the normalized world mover's distance to represent a value between 0.0 and 1.0 as closeness.

# Code Execution

To run this script, please execute the following command:

`python3 code/complete_relevance_matrix.py --input data/RELISH_tokenized.npy --matrix data/relevance_WMD.tsv`