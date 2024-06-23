# Part of Speech Tagger using Feed Forward Neural Net (FFNN) and Recurrent Neural Net (RNN)

## Overview
This repository contains code for a Part-of-Speech (POS) tagger implemented using a Feed Forward Neural Network (FFNN) and a Recurrent Neural Network (RNN). The POS tagger is trained on Universal Dependencies datasets (en_atis-ud) and evaluated using test datasets. The implementation includes data loading, preprocessing, model training, evaluation, and inference functionalities. 

## Files
* main.py: Contains the main script to execute the POS tagging pipeline.
* ffnn.py: Defines the FFNN model architecture (FFNN class).
* rnn.py: Defines the RNN model architecture

  ## FFNN Architecture
  * The  consists of an embedding layer followed by two fully connected layers (fc1 and fc2). The size of the input to fc1 is determined by the concatenation of embeddings of p + s context words, where 'p' denotes preceding context and 's' denotes succeeding context.

## Execution
FFNN -> python3 pos_tagger.py -f 
LSTM -> pythgon3 pos_tagger.py -r

