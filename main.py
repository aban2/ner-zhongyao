from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from model_processor import Processor
import bertcrf


train, valid, test = divide_dataset(get_train_data())

processor = Processor()#load=200)#train=train)
print(processor)

# args = {
# 	is_train=1,
# 	batch_size=16,
# 	save_epoch=100,
# 	max_norm=max_grad_norm=1.0
# }

is_train = 0

if is_train:
	processor.train(train=train, valid=valid, test=test, num_epoches=1, batch_size=1, save_epoch=100)
else:
	# print(processor.evaluate(test, '325'))
	ct = 1
	for i in range(1000, 1500):
		print(i)
		name = str(i)
		t = processor.predict(filename=name, epoch='600')

		with open('train/'+name+'.ann', 'w', encoding='utf-8') as f:
			f.write(t)

		# print(t)

		ct += 1

		# break

		if t == None:
			print('oh')
			break