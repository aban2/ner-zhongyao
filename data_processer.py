import numpy as np
from data_loader import get_train_data
import pickle

def preprocessing(data_list):
	para2ints = {}
	para_lengths = []
	mx = 0
	for idx, data in enumerate(data_list):
		paragraph, tuples = data

		#if len(paragraph) > 1000:
			# print(paragraph)
			# print(len(paragraph))
			# print(idx)

		last = 0
		for idx2, word in enumerate(paragraph):
			if word == ' ' or word == '。':
				mx = max(mx, idx2-last)
				last = idx2

	print(mxs)

		#print('hi')
			#break

		# process label
		# idx2label = {}
		# for tup in tuples:
		# 	_, label, start, end, name = tup
		# 	for i in range(start, end+1):
		# 		idx2label[i] = label

		# # process paragraph
		# offset = 0
		# for word in sent:
		# 	if word != ' ' and word != '.':
		# 		sent_list.append(word)
		# sents.append(sent_list)
	#return sents
	return 0


def divide_dataset(data):
	np.random.seed(0)
	data_list = []
	for i in range(len(data)):
		data_list.append(data[i])
	data_list = np.array(data_list)
	np.random.shuffle(data_list)

	#data_list = preprocessing(data_list)

	return data_list[0:900], data_list[900:950], data_list[950:]

def data2pixel(data):
	with open('data', 'wb') as f:
		pickle.dump(data, f)

if __name__ == '__main__':
	for i in range(10, 15):
		print(i)