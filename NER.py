import pandas as pd
import numpy as np
import scipy as sp
from seqlearn.datasets import load_conll 
from seqlearn.hmm import MultinomialHMM
from seqlearn.evaluation import bio_f_score






#Fill this function with features to train seqlearn on.
#This is where the bulk of the work is to optimize the implementation
def features(sequence, i):
	yield "word=" + sequence[i].lower()
	if sequence[i].isupper():
		yield "Uppercase"








def main():
	#Load in training data and pass it through our feature function.
	#See documentation exact outputs of load_conll
	samples, labels, sentence_lengths = load_conll("data/gene-trainF18.txt", features)
	


	#Train the model with our features
	hmm = MultinomialHMM()
	hmm.fit(samples, labels, sentence_lengths)


	#Evaluate our model
	test_samples, test_labels, test_sentence_lengths = load_conll("data/test-run-test-with-keys.txt", features)
	prediction = hmm.predict(test_samples, test_sentence_lengths)
	print(bio_f_score(test_labels, prediction))









if __name__ == '__main__':
	main()