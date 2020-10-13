import sys
import torch
from torch import nn
from utils import *
from allennlp.modules import ConditionalRandomField
from transformers import BertForTokenClassification

class BERTCRF(nn.Module):

	def __init__(self, args):
		super(BERTCRF, self).__init__()
		if args['load_model'] <= 0:
			self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(label2id))

		else:
			self.model = torch.load('models/Mod' + str(load))
			print('load success')

		self.crf = ConditionalRandomField(len(label2id))

	def forward(self, ids, masks, labels, mode='train'):
		loss, logits = self.model(ids, attention_mask=masks, labels=labels)
		if mode == 'train':
			out = -self.crf(logits, labels, masks)
			return out
		else:
			tups = self.crf.viterbi_tags(logits, masks)
			out = []
			for tup in tups:
				out.append(tup[0])
			return loss, out
