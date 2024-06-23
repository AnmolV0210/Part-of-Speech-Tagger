# Part of Speech Tagger using Feed Forward Neural Net (FFNN) and Recurrent Neural Net (RNN)

## Overview
This repository contains code for a Part-of-Speech (POS) tagger implemented using a Feed Forward Neural Network (FFNN) and a Recurrent Neural Network (RNN). The POS tagger is trained on Universal Dependencies datasets (en_atis-ud) and evaluated using test datasets. The implementation includes data loading, preprocessing, model training, evaluation, and inference functionalities. 

## Files
* main.py: Contains the main script to execute the POS tagging pipeline.
* ffnn.py: Defines the FFNN model architecture (FFNN class).
* rnn.py: Defines the RNN model architecture

  ## Model Architecture
  ### FFNN
  * The  consists of an embedding layer followed by two fully connected layers (fc1 and fc2). The size of the input to fc1 is determined by the concatenation of embeddings of p + s context words, where 'p' denotes preceding context and 's' denotes succeeding context.

 ### RNN
 * The model consists of an embedding layer, an LSTM layer, and a linear layer for tag prediction. The model uses log-softmax for output and is designed to handle variable-length sequences with batch_first=True.

## Dataset
### Universal Dependencies Dataset
The Universal Dependencies dataset is a collection of syntactically annotated corpora in multiple languages, designed to facilitate research in natural language processing (NLP) and linguistic analysis. It provides linguistically motivated, cross-linguistically consistent treebanks, with annotations for parts of speech (POS), syntactic dependencies, and morphological features.
    

## Execution
FFNN -> python3 pos_tagger.py -f 
LSTM -> python3 pos_tagger.py -r

