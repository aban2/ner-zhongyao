import numpy as np
from data_loader import get_train_data

def divide_dataset(data):
	np.random.seed(0)
	data_list = []
	for i in range(len(data)):
		data_list.append(data[i])
	data_list = np.array(data_list)
	np.random.shuffle(data_list)

	return data_list[0:900], data_list[900:950], data_list[950:]

if __name__ == '__main__':
	divide_dataset(get_train_data())