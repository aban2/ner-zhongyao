# -*- coding:utf-8 -*-
import sys
import torch
import numpy as np
from time import time
from data_processer import padding, split_data
from utils import get_label_dic, load_label_dic
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

class Processor:
	def __init__(self, load=0, train=None):
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
		# print(self.id2label)
		if load == 0:
			self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(self.label2id)).to(self.device)
		else:
			self.model = torch.load('models/Mod' + str(load))
			print('load success')
		self.epoch_ct = load


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

		# torch.save(optimizer, 'models/Opt' + str(0+self.epoch_ct+1))
		# torch.save(scheduler, 'models/Sch' + str(0+self.epoch_ct+1))

		# sys.exit()

		# training
		top = 1.3
		start_time = time()
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

			F0 = None
			if (i+1) % 5 == 0:
				F0, _ = self.evaluate(train)
			F1, loss = self.evaluate(valid)
			F2, loss2 = self.evaluate(test)

			if F1+F2 > top:
				top = F1 + F2
				torch.save(self.model, 'models/Mod' + str(i+self.epoch_ct+1))
				print('save new top', top)

			print('Epoch', i+self.epoch_ct, losses/len(train_dataloader), loss, 'F1', F1, F2, F0, time()-start_time)

			# if (i+1) % save_epoch == 0:
			torch.save(self.model, 'models/Mod' + str(i+self.epoch_ct+1))
			start_time = time()

	def evaluate(self, valid, epoch=None):
		# get dataloader
		batch_size = 64
		dataloader, followed = self.data2loader(valid, 'valid', batch_size=batch_size)

		if epoch != None:
			model = torch.load('models/Mod' + epoch)
		else:
			model = self.model

		model.eval()
		with torch.no_grad():
			losses = 0
			recalls, accs = [], []

			# process all data in one batch
			F1s, losses = [], []
			last_correct, last_pred, last_true, last_F1 = 0, 0, 0, 0
			crfs = 0
			# print(len(followed), np.sum(followed))
			for idx, batch_data in enumerate(dataloader):
				# print(idx)
				batch_data = tuple(i.to(self.device) for i in batch_data)
				ids, masks, labels = batch_data
				loss, logits = model(ids, attention_mask=masks, labels=labels) # loss and logits

				losses.append(loss.item())

				results = torch.argmax(logits, dim=2)
				results = results*masks
				# print(results.shape)

				for mdx, result in enumerate(results):
					lenth = torch.sum(masks[mdx]).item()

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

					if followed[mdx+idx*batch_size] == 0:
						if mdx+idx > 0:
							# print(mdx+idx*batch_size)
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

					# 	break
					# break

			F1s.append(last_F1)
			return np.mean(F1s), np.mean(losses)
	def predict(self, filename, epoch):
		# read
		with open('chusai_xuanshou/'+filename+'.txt', 'r', encoding='utf-8') as f:
			content = f.read()
		model = torch.load('models/Mod' + epoch)

		# predict
		space = ','
		content = content.replace(' ', space).replace('　', space)
		content = list(content)
		if len(content) > 510:
			data_list = split_data(content)
		else:
			data_list = [content]

		# print(data_list)

		ret_str = ''
		ct = 1
		model.eval()
		para_offset = 0
		offsets = []
		space_ct = -1

		# calculate offset
		for c in content:
			if c == ' ' or c == '　':
				space_ct += 1
			else:
				offsets.append(space_ct)

		stop_words = ['.', ',', '(', ')', '。', '，','、', '（','）', ':', '：']
		crfs = 0
		with torch.no_grad():
			for kdx, data in enumerate(data_list):
				t = self.tokenizer(data, is_split_into_words=True, return_tensors='pt')
				_, logits = model(t['input_ids'].to(self.device), attention_mask=t['attention_mask'].to(self.device), labels=t['token_type_ids'].to(self.device))
				result = torch.argmax(logits, dim=2)
				result = torch.squeeze(result, 0)
				# extract entities
				record = -1
				record_pos = -1
				entity = ""

				for idx, c in enumerate(result):
					word = self.tokenizer.decode(t['input_ids'][0][idx].item())
					if record < 0 and c > 1 and c & 1:
						entity += word
						record = c-1
						record_pos = idx
					elif c == record:
						entity += word
					elif record > 1 and c != record:
						# check crf:
						# if c > 1 and (c & 1 == 0):
						# 	print(result)
						# 	crfs = 1
						# 	print('wrong crf')
						# 	return

						extra = '\n'
						if ret_str == '':
							extra = ''
						offset = offsets[record_pos+para_offset] + para_offset
						new_start, new_end = record_pos+offset, idx+offset
						real_entity = ''.join(content[new_start:new_end])

						# truncate
						truc_entity = ""
						truncation = -1
						for qdx, q in enumerate(real_entity):
							if q in stop_words:
								truncation = qdx
								break
						if truncation > 0:
							new_end = new_start+truncation
							real_entity = real_entity[0:truncation]

						# if truncation > 0:
							# print('hi')
						ret_str += extra + 'T' + str(ct) + '\t' + self.id2label[record.item()][2:] + ' ' + str(new_start) + ' ' + str(new_end) + '\t' + real_entity
						if truncation < 0 and entity != real_entity and ' ' not in real_entity and '　' not in real_entity and '[ U N K ]' not in entity:
							print('wrong', filename)
							print(entity)
							print(real_entity)
							return
						# print(new_start, new_end, offset, ''.join(content[new_start:new_end]))
						# reset
						ct += 1
						if c > 1 and c & 1:
							record_pos = idx
							record = c-1
							entity = word
						else:
							record_pos = -1
							record = -1
							entity = ''
				para_offset += result.shape[0]-2

		# print(ret_str)
		return ret_str

		# write ann

if __name__ == '__main__':
	filename = '1000'
	with open('chusai_xuanshou/'+filename+'.txt', 'r', encoding='utf-8') as f:
		content = f.read()

	print(content)