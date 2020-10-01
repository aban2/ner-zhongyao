import numpy as np
from data_loader import get_train_data
import pickle
from time import time

def padding(data, tokenizer, label2id):
	data, labels = data[:,0], data[:,1]

	data = list(data)
	start_time = time()

	t = tokenizer(data, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')

	print(time() - start_time)

	print(type(data))

	# process data
	padded_data = []
	for idx, sample in enumerate(data):
		padded_sent = tokenizer(sample, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
		padded_data.append(padded_sent)

	# process labels
	for tuples in labels:
		for tup in tuples:
			_, label, start, end, name = tup


	return padded_data, 0

def divide_dataset(data):
	np.random.seed(0)
	data_list = []
	for i in range(len(data)):
		data_list.append(data[i])
	data_list = np.array(data_list)
	np.random.shuffle(data_list)

	return data_list[0:900], data_list[900:950], data_list[950:]

def data2pixel(data):
	with open('data', 'wb') as f:
		pickle.dump(data, f)


if __name__ == '__main__':
	for i in range(10, 15):
		print(i)