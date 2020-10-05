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
		self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(self.label2id)).to(self.device)

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

	def train(self, train, valid, test, num_epoches, batch_size, save_epoch, max_grad_norm=1.0):
		# get dataloader
		train_dataloader, _ = self.data2loader(train, mode='train', batch_size=batch_size)

		# optimizer and scheduler
		FULL_FINETUNING = True
		if FULL_FINETUNING:
		    param_optimizer = list(self.model.named_parameters())
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
		top = 1.6
		for i in range(num_epoches):
			self.model.train()

			losses = 0
			for idx, batch_data in enumerate(train_dataloader):
				#print(idx)
				batch_data = tuple(i.to(self.device) for i in batch_data)
				ids, masks, labels = batch_data

				self.model.zero_grad()
				loss, _ = self.model(ids, attention_mask=masks, labels=labels)

				# process loss
				loss.backward()
				losses += loss.item()

				# tackle exploding gradients
				torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

				optimizer.step()
				scheduler.step()

				# if (idx+1) % 100 == 0:
				# 	print('batch', idx+1, 'loss', losses/(idx+1))

			F1, loss = self.evaluate(valid)
			F2, loss2 = self.evaluate(test)

			if F1+F2 > top:
				top = F1 + F2
				torch.save(self.model, 'models/Mod' + str(i+1))
				print('save new top', top)

			print('Epoch', i, losses/len(train_dataloader), loss, 'F1', F1, F2)

			if (i+1) % save_epoch == 0:
				torch.save(self.model, 'models/Mod' + str(i+1))

		return model

	def evaluate(self, valid, epoch=None):
		# get dataloader
		dataloader, followed = self.data2loader(valid, 'valid', batch_size=1024)

		if epoch != None:
			model = torch.load('models/Mod' + epoch)
		else:
			model = self.model

		print()

		model.eval()
		# print(model)

		with torch.no_grad():
			losses = 0
			recalls, accs = [], []

			# process all data in one batch
			F1s = []
			last_correct, last_pred, last_true, last_F1 = 0, 0, 0, 0
			# print(len(followed), np.sum(followed))
			for idx, batch_data in enumerate(dataloader):
				batch_data = tuple(i.to(self.device) for i in batch_data)
				ids, masks, labels = batch_data
				loss, logits = model(ids, attention_mask=masks, labels=labels) # loss and logits

				results = torch.argmax(logits, dim=2)
				results = results*masks


				for mdx, result in enumerate(results):

					lenth = torch.sum(masks[mdx]).item()
					# print(labels[idx][0:lenth+1])
					# print()
					# print(result[0:lenth+1])
					# print()

					channel_pred, channel_true = -1, -1
					following, correct, total_pred, total_true = -1, 0, 0, 0
					for jdx, (label_pred, label_true) in enumerate(zip(result, labels[mdx])):
						label_true, label_pred = label_true.item(), label_pred.item()

						# process total true
						if label_true != channel_true:
							if following == 1 and label_pred != channel_pred:
								correct += 1
								# print(channel_pred, channel_true)
							following = 0

						if label_true & 1 and label_true > 1:
							channel_true = label_true - 1
							total_true += 1
						else:
							channel_true = label_true

						# process total preds
						if label_pred & 1 and label_pred > 1:
							total_pred += 1
							channel_pred = label_pred - 1
							if following == 0 and label_pred == label_true:
								following = 1
						else:
							channel_pred = label_pred

						if following == 1 and label_pred != label_true:
							following = -1

						if label_true == 0:
							break

					if total_pred == 0:
						precision = 0
					else:
						precision = correct / total_pred
					if total_true == 0:
						recall = 0
					else:
						recall = correct / total_true
					if precision + recall == 0:
						F1 = 0
					else:
						F1 = 2*(precision*recall) / (precision+recall)

					if followed[mdx] == 0:
						if mdx > 0:
							F1s.append(last_F1)
						last_correct, last_pred, last_true, last_F1 = correct, total_pred, total_true, F1
					else:
						last_correct += correct
						last_true += total_true
						last_pred += total_pred
						if last_pred == 0:
							precision = 0
						else:
							precision = last_correct / last_pred
						if last_true == 0:
							recall = 0
						else:
							recall = correct / last_true
						if precision + recall == 0:
							last_F1 = 0
						else:
							last_F1 = 2*(precision*recall) / (precision+recall)

				F1s.append(last_F1)
				# print(len(F1s))

			# print('Evaluation: F1', np.mean(F1s), 'loss', loss)
			return F1, loss.item()