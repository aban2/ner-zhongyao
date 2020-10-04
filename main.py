from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from processor import Processor

train, valid, test = divide_dataset(load_pickle())

processor = Processor()#train=train)

is_train = 1

if is_train:
	processor.train(train=train, num_epoches=100, max_grad_norm=1.0, batch_size=16, save_epoch=20)
else:
	processor.evaluate('601', valid)