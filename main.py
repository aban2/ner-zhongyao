from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from processor import Processor

train, valid, test = divide_dataset(load_pickle())

processor = Processor()#train=train)

is_train = 0
continue_train = 1

if is_train:
	processor.train(train=train, valid=valid, test=test, num_epoches=100, batch_size=1, save_epoch=20)
else:
	processor.evaluate(valid, '80')