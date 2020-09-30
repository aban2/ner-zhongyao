import numpy as np
from data_loader import get_train_data
import pickle

def preprocessing(data_list):
	sents, labels = [], []

	for data in data_list:
		sent_list = []
		sent, label = data
		for word in sent:
			if word != ' ' and word != '.':
				sent_list.append(word)
		sents.append(sent_list)
	return sents


def divide_dataset(data):
	np.random.seed(0)
	data_list = []
	for i in range(len(data)):
		data_list.append(data[i])
	data_list = np.array(data_list)
	np.random.shuffle(data_list)

	q = preprocessing(data_list)

	return q[0:900], q[900:950], q[950:]

def data2pixel(data):
	with open('data', 'wb') as f:
		pickle.dump(data, f)

if __name__ == '__main__':
	# divide_dataset(get_train_data())
	data2pixel(get_train_data())