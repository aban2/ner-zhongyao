import sys
import torch
import pickle
import numpy as np
from time import time
from data_loader import get_train_data

def split_data(data):
	data_list = []
	stop_list = [' ', '。', '　']
	start = 0
	tag = 510
	i = tag

	while True:
		if data[i] in stop_list:
			data_list.append(data[start:i])
			if i + 505 > len(data):
				data_list.append(data[i:len(data)])
				break	
			tag = i + 510
			start = i
			i = tag
		i -= 1

	return data_list

def padding(data, tokenizer, label2id):
	# init ds
	follow_indices = []
	data_list_to_pad = []
	labels_list_to_pad = []

	for tup in data:
		# print(tup)
		# print('new data!')
		data, labels = tup

		if len(data) > 510:
			data_list = split_data(data)
		else:
			data_list = [data]

		# process label
		start2endlab = {}
		for tup in labels:
			_, label, start, end, name = tup
			start2endlab[int(start)] = (int(end), label, name)

		offset = 0
		for mdx, data in enumerate(data_list):
			# process data
			data = list(data)

			labs = [label2id['O']]
			yes_end = -1
			yes_lab = -1

			# correctness checking
			for idx, c in enumerate(data):
				if c == ' ' or c == '　':
					continue

				if idx == yes_end:
					yes_end = -1
					yes_lab = -1

				if (idx + offset) in start2endlab:
					yes_end, yes_lab, name = start2endlab[idx+offset]
					name2 = ''.join(data[idx:yes_end-offset])
					if name != name2:
						print('checking failed')
						print(name, name2)
					labs.append(label2id['B_'+yes_lab])

				elif yes_end > 0:
					labs.append(label2id['I_'+yes_lab])

				else:
					labs.append(label2id['O'])
					
			offset += len(data)

			labs.append(label2id['O'])
			tk = tokenizer(data, is_split_into_words=True, truncation=True)

			# print(labs)
			# print(labels)
			# sys.exit()

			# print(tk)

			if len(labs) != len(tk['input_ids']):
				print("wrong match", len(labs), len(tk['input_ids']))

			data_list_to_pad.append(data)
			labels_list_to_pad.append(labs)
			if mdx == 0:
				follow_indices.append(0)
			else:
				follow_indices.append(1)

	# print(labels_list_to_pad)

	padded_data = tokenizer(data_list_to_pad, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
	longest_len = padded_data['input_ids'].shape[1]
	padded_labels = []

	# print(labs)

	for idx, lab in enumerate(labels_list_to_pad):
		lab.extend([label2id['[PAD]']]*(longest_len-len(lab)))
		padded_labels.append(lab)

	padded_labels = torch.tensor(padded_labels)

	return padded_data, padded_labels, follow_indices

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
	t = 5
	for i in range(t):
		if t > 3:
			t = 3
		print(i)