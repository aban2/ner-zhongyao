from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from model_processor import Processor
from time import time
import pickle
import bertcrf
import os
import sys

fold = 2
train, valid, test = divide_dataset(load_pickle(), fold=fold)

args = {
	'load_model':0,
	'num_epoches': 1,
	'save_epoch': 1,
	'fold': fold,

	'is_train': 0,
	'batch_size': 1,
	'max_grad_norm': 1.0,
	'train':train,
	'valid':valid,
	'test':test
}

processor = Processor(args)

if args['is_train'] > 0:
	processor.train()
else:
	# get model list
	models = []
	fileList = os.listdir('models')
	for idx, file in enumerate(fileList):
		if '_300' in file:
			models.append(file)

	print(models)

	id2dict = {}

	for i in range(1000, 1500):
		start_time = time()

		filename = str(i)

		ret_dic = {}
		for model in models:
			_, ret_dic = processor.predict(filename, ret_dic, model)
			# break
			# print('hi')

		id2dict[i] = ret_dic

		# with open('train/'+filename+'.ann', 'w', encoding='utf-8') as f:
		# 	f.write(t)

		# for key in ret_dic:
		# 	print(key ,ret_dic[key])
		# print()
		print(i, time()-start_time)

		# break
		if ret_dic == None:
			print('oh')
			break

	with open('300_dic.pkl', 'wb') as f:
		pickle.dump(id2dict, f)