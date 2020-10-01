def get_label_dic(train_labels):
	id2label = {0:'N', 1:'[PAD]'}
	label2id = {'N':0, '[PAD]':1}
	ct = 2

	for tuples in train_labels:
		for tup in tuples:
			_, label, _, _, _ = tup
			if label not in label2id:
				label2id[label] = ct
				id2label[ct] = label
				ct += 1
	return id2label, label2id