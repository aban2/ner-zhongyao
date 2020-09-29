from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

train, valid, test = divide_dataset(load_pickle())

# get ints
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_padded = tokenizer(train_sentences, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
valid_padded = tokenizer(valid_sentences, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')

# a change