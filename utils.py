import pickle
import torch

# get device
if torch.cuda.is_available():
	device = torch.device('cuda')
	name = torch.cuda.get_device_name(0)
else:
	device = torch.device('cpu')
	name = 'Cpu'
print('Running On', name)

def get_label_dic(train_labels):
	id2label = {0:'[PAD]', 1:'O'}
	label2id = {'[PAD]':0, 'O':1}
	ct = 2

	for tuples in train_labels:
		for tup in tuples:
			_, label, _, _, _ = tup
			if ('I_'+label) not in label2id:
				# IN
				label2id['I_'+label] = ct
				id2label[ct] = 'I_'+label
				ct += 1

				# BEGIN
				label2id['B_'+label] = ct
				id2label[ct] = 'B_'+label
				ct += 1

	with open('label_id_dic', 'wb') as f:
		print('save_label_id_dic')
		pickle.dump((label2id, id2label), f)

	return id2label, label2id

def load_label_dic():
	with open('label_id_dic', 'rb') as f:
		content = pickle.load(f)
	return content

label2id, id2label = load_label_dic()

if __name__ == '__main__':

	blade = 5

	with open('final_dic.pkl', 'rb') as f:
		dic = pickle.load(f)

	# check f
	# dics = {}
	# for i in dic:
	# 	dic2 = dic[i]
	# 	for j in dic2:
	# 		if dic2[j] in dics:
	# 			dics[dic2[j]].append(j[3])
	# 		else:
	# 			dics[dic2[j]] = [j[3]]

	# sums = 0
	# for t in dics:
	# 	# print(t, len(dics[t]))
	# 	if t >= 6:
	# 		sums += len(dics[t])
	
	# print(sums/500)

	for i in dic:
		with open('train/'+str(i)+'.ann', 'w', encoding='utf-8') as f:
			dic2 = dic[i]
			ct = 1
			for idx, j in enumerate(dic2):
				if dic2[j] < blade:
					continue
				if idx == 0:
					extra = ''
				else:
					extra = '\n'
				t = extra + 'T' + str(ct) + '\t' + j[0] + ' ' + str(j[1]) + ' ' + str(j[2]) + '\t' + j[3]
				f.write(t)
				ct += 1