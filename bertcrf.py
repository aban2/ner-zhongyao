import sys
import torch
from torch import nn
from utils import *
from allennlp.modules import ConditionalRandomField
from transformers import BertForTokenClassification

class BERTCRF(nn.Module):

	def __init__(self, use_crf):
		super(BERTCRF, self).__init__()
		self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(label2id))
		self.crf = ConditionalRandomField(len(label2id), include_start_end_transitions=False)
		if use_crf == False:
			self.crf.transitions.data.fill_(0)
			self.crf.transitions.requires_grad = False

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