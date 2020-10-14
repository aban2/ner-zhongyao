from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from model_processor import Processor
import bertcrf
import sys

train, valid, test = divide_dataset(load_pickle())
args = {
	'load_model':50,
	'num_epoches': 3,
	'save_epoch': 2,

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
	for i in range(1000, 1500):
		print(i)
		filename = str(i)
		t = processor.predict(filename)

		# with open('train/'+name+'.ann', 'w', encoding='utf-8') as f:
		# 	f.write(t)

		print(t)

		# break
		if t == None:
			print('oh')
			break