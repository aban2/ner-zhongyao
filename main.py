from data_loader import get_train_data, load_pickle
import data_processer
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

train, valid, test = data_processer.divide_dataset(load_pickle())
# print(train)


# get ints
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

padded_train = []

for idx, sample in enumerate(train[:,0]):
	print(sample)
	padded_sent = tokenizer(sample)
	padded_train.append(padded_sent)
	print(padded_sent)
	break


# a change