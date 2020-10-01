from data_loader import get_train_data, load_pickle
from utils import get_label_dic
import data_processer
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys

train, valid, test = data_processer.divide_dataset(load_pickle())
id2label, label2id = get_label_dic(train[:,1])

# get padded_data
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
padded_train_data, padded_train_labels = data_processer.padding(train, tokenizer, label2id)