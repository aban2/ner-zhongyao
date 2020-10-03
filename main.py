import sys
import torch
from utils import get_label_dic
from data_loader import get_train_data, load_pickle
from data_processer import divide_dataset, padding
from transformers import BertTokenizer, BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train, valid, test = divide_dataset(load_pickle())
id2label, label2id = get_label_dic(train[:,1])

# get padded_data
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

padded_train_data, padded_train_labels, _ = padding(train, tokenizer, label2id)
padded_valid_data, padded_valid_labels, _ = padding(valid, tokenizer, label2id)

print(padded_train_data['input_ids'].shape, padded_train_labels.shape)
print(padded_valid_data['input_ids'].shape, padded_valid_labels.shape)

batch_size = 16
num_epoches = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_data = TensorDataset(padded_train_data['input_ids'], padded_train_data['attention_mask'], padded_train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

valid_data = TensorDataset(padded_valid_data['input_ids'], padded_valid_data['attention_mask'], padded_valid_labels)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

# model
model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(label2id)).to(device)

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
		batch_data = tuple(i.to(device) for i in batch_data)
		ids, masks, labels = batch_data
		#print(ids.shape, masks.shape, labels.shape)

		model.zero_grad()
		output = model(ids, attention_mask=masks, labels=labels)

		# process loss
		loss = output[0]
		loss.backward()
		losses += loss.item()

		# tackle exploding gradients
		torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

		optimizer.step()
		scheduler.step()

		if (idx+1) % 100 == 0:
			print('batch', idx+1, 'loss', losses/(idx+1))

	model.eval()
	# Reset the validation loss for this epoch.
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	predictions , true_labels = [], []
	for batch in valid_dataloader:
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_labels = batch

		# Telling the model not to compute or store gradients,
		# saving memory and speeding up validation
		with torch.no_grad():
			# Forward pass, calculate logit predictions.
			# This will return the logits rather than the loss because we have not provided labels.
			outputs = model(b_input_ids, token_type_ids=None,
				attention_mask=b_input_mask, labels=b_labels)
		# Move logits and labels to CPU
		logits = outputs[1].detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()

		# Calculate the accuracy for this batch of test sentences.
		eval_loss += outputs[0].mean().item()
		predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
		true_labels.extend(label_ids)

	eval_loss = eval_loss / len(valid_dataloader)
	validation_loss_values.append(eval_loss)
	print("Validation loss: {}".format(eval_loss))
	pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
					for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
	valid_tags = [tag_values[l_i] for l in true_labels
						for l_i in l if tag_values[l_i] != "PAD"]

	print('Epoch', i+1, losses/len(train))