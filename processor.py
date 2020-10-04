import sys
import torch
import numpy as np
from data_processer import padding
from utils import get_label_dic, load_label_dic
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

class Processor:

	def __init__(self, train=None):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
		if train != None:
			self.label2id, self.id2label = get_label_dic(train[:,1])
		else:
			self.label2id, self.id2label = load_label_dic()
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			name = torch.cuda.get_device_name(0)
		else:
			self.device = torch.device('cpu')
			name = 'Cpu'
		print('Running On', name)

	def data2loader(self, data, mode, batch_size):
		padded_data, padded_labels, followed = padding(data, self.tokenizer, self.label2id)
		data = TensorDataset(padded_data['input_ids'], padded_data['attention_mask'], padded_labels)
		if mode == 'train':
			sampler = RandomSampler(data)
			# print('sui')
		else:
			sampler = SequentialSampler(data)
			# print('busui')
		dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
		return dataloader, followed

	def train(self, train, num_epoches, max_grad_norm, batch_size):
		# get dataloader
		train_dataloader, _ = self.data2loader(train, mode='train', batch_size=batch_size)

		# model
		model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(self.label2id)).to(self.device)

		# optimizer and scheduler
		FULL_FINETUNING = True
		if FULL_FINETUNING:
		    param_optimizer = list(model.named_parameters())
		    no_decay = ['bias', 'gamma', 'beta']
		    optimizer_grouped_parameters = [
		        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		         'weight_decay_rate': 0.01},
		        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
		         'weight_decay_rate': 0.0}
		    ]
		else:
		    param_optimizer = list(model.classifier.named_parameters())
		    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

		optimizer = AdamW(
		    optimizer_grouped_parameters,
		    lr=3e-5,
		    eps=1e-8
		)

		total_steps = len(train_dataloader) * num_epoches
		scheduler = get_linear_schedule_with_warmup(
		    optimizer,
		    num_warmup_steps=0,
		    num_training_steps=total_steps
		)

		# training
		for i in range(num_epoches):
			model.train()

			losses = 0
			for idx, batch_data in enumerate(train_dataloader):
				#print(idx)
				batch_data = tuple(i.to(self.device) for i in batch_data)
				ids, masks, labels = batch_data

				model.zero_grad()
				loss, _ = model(ids, attention_mask=masks, labels=labels)

				# process loss
				loss.backward()
				losses += loss.item()

				# tackle exploding gradients
				torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

				optimizer.step()
				scheduler.step()

				if (idx+1) % 100 == 0:
					print('batch', idx+1, 'loss', losses/(idx+1))

			if (i+1) % 25 == 0:
				torch.save(model, 'Mod' + str(i+1))	

		return model

	def evaluate(self, epoch, valid):
		# get dataloader
		dataloader, followed = self.data2loader(valid, 'valid', batch_size=512)

		model = torch.load('models/Mod' + epoch)
		model.eval()
		# print(model)

		with torch.no_grad():
			losses = 0
			recalls, accs = [], []

			# process all data in one batch
			for idx, batch_data in enumerate(dataloader):
				batch_data = tuple(i.to(self.device) for i in batch_data)
				ids, masks, labels = batch_data
				loss, logits = model(ids, attention_mask=masks, labels=labels) # loss and logits
				print(loss)

				results = torch.argmax(logits, dim=2)
				results = results*masks

				for idx, result in enumerate(results):

					lenth = torch.sum(masks[idx]).item()
					print(labels[idx][0:lenth+1])
					print()
					print(result[0:lenth+1])
					print()
					channel_true, channel_pred = -1, -1
					following, correct, total_preds, total_true = 0, 0, 0, 0

					for jdx, (label_pred, label_true) in enumerate(zip(result, labels[idx])):
						label_true, label_pred = label_true.item(), label_pred.item()

						# process channel_true
						if channel_true > 1:
							if label_true != channel_true:
								if label_true > 1:
									total_true += 1
								if following == 1 and label_pred != channel_pred:
									correct += 1
								following = 0
						else:
							if label_true > 1:
								total_true += 1
								following = 0

						# process channel_pred
						if channel_pred > 1:
							if channel_pred != label_pred:
								if label_pred > 1:
									total_preds += 1
									if following == 0:
										following = 1

							if following == 1:
								if label_pred != label_true:
									following = -1
						else:
							if label_pred > 1:
								total_preds += 1
								if following == 0:
									if label_pred == label_true:
										following = 1
									else:
										following = -1

						channel_true = label_true
						channel_pred = label_pred

						if label_true == 0:
							break

					print(correct, total_preds, total_true)

					
					if idx == 1:
						break
