from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

train, valid, test = divide_dataset(load_pickle())
print(train)


# get ints
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

padded_sents = tokenizer(train)

# a change