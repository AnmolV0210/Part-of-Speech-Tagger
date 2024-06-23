# ASSIGNMENT2 

* The POS tagging for input sentence function is not implemented in pos_tagger.py. Instead I have written it indiviudally in ffnn.py and rnn.py 
* The main() functions from either of the files is called, depending on the argument -f or -r
* Validation accuracy vs Epoch graph is not plotted for FFNN.
* Precision, recall, f1 score is not implemented.
* I have included intermediate .ipynb files for both rnn and ffnn for generating graphs and to show hyperparameter tuning 

## Execution
FFNN -> python3 pos_tagger.py -f 
LSTM -> pythgon3 pos_tagger.py -r

