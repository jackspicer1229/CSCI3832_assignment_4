import re
from seqlearn.datasets import load_conll
from seqlearn.hmm import MultinomialHMM
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.evaluation import bio_f_score


### Jack Spicer
### Sousheel Vunnam
### Nikhil Rajaram

#Fill this function with features to train seqlearn on.
#This is where the bulk of the work is to optimize the implementation
def features(sequence, i):
	# word = sequence[i].split()[1]
	word = sequence[i][1]
	yield "word=" + word.lower()
	if word.isupper():
		yield "Uppercase"

	#Check if word i contains a number
	if re.search('.*\d.*', word):
		yield "Numeric"

	#Check if word contains a capital letter, but not the first letter
	if re.search('\b[a-z]+[A-Z]+[a-z]*', word):
		yield "Capital"
	if i > 0:
		yield "word-1:{}" + sequence[i - 1][1].lower()
		if i > 1:
			yield "word-2:{}" + sequence[i - 2][1].lower()
	if i + 1 < len(sequence):
		yield "word+1:{}" + sequence[i + 1][1].lower()
		if i + 2 < len(sequence):
			yield "word+2:{}" + sequence[i + 2][1].lower()

	yield str(len(word))

	if re.search('[.!?\\-]', word):
		yield re.findall('[.!?\\-]', word)[0]





def main():
	#Load in training data and pass it through our feature function.
	#See documentation exact outputs of load_conll
	samples, labels, sentence_lengths = load_conll("data/gene-trainF18.txt", features, split=True)


	#Train the model with our features
	clf = StructuredPerceptron()
	clf.fit(samples, labels, sentence_lengths)


	#Evaluate our model
	test_samples, test_labels, test_sentence_lengths = load_conll("data/F18-assgn4-test.txt", features, split=True)
	prediction = clf.predict(test_samples, test_sentence_lengths)

	#Output results
	i = 0
	j = 1
	output = []
	for line in open("data/F18-assgn4-test.txt"):
		if(line == "\n"):
			output.append("\n")
			j = 1
			continue
		else:
			item = str(j) + "\t" + line.split()[1] + "\t" + prediction[i] + "\n"
			output.append(item)
			print(item)
			i+=1
			j+=1

	with open('predictions.txt', 'w') as f:
		for item in output:
			f.write(item)


	#Add dummy tag to the end of data
	# file_edit = []
	# with open('data/F18-assgn4-test.txt') as fp:
	# 	line = fp.readline()
	# 	while line:
	# 		if line == "\n":
	# 			file_edit.append("\n")
	# 			line = fp.readline()
	# 			continue
	# 		else:
	# 			file_edit.append(line[:len(line)-1] + "\t" + "O" + "\n")
	# 			line = fp.readline()

	# with open('data/F18-assgn4-test-1.txt', 'w') as f:
	# 	for item in file_edit:
	# 		f.write(item)





if __name__ == '__main__':
	main()
