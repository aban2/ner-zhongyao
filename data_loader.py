import os
import pandas as pd
import numpy as np
import pickle

def get_train_data():
	id2input = {}
	path = 'train'
	fileList = os.listdir(path)

	for idx, file in enumerate(fileList):
		name, suffix = os.path.splitext(file)
		name = int(name)
		entities = []
		with open(path+'/'+file, encoding='utf-8') as f:
			lines = f.read().splitlines()
			if suffix == '.txt':
				if name in id2input:
					id2input[name][0] = lines[0]
				else:
					id2input[name] = [lines[0], None]
			else:
				for line in lines:
					entity = []
					s1 = line.split('\t')
					mid = s1[1].split(' ')
					entity.append(s1[0])
					entity.extend(mid)
					entity.append(s1[2])
					entities.append(entity)
				if name in id2input:
					id2input[name][1] = entities
				else:
					id2input[name] = [None, np.array(entities)]

	for i in id2input:
		text, entities = id2input[i]
		for entity in entities:
			start, end, name = int(entity[2]), int(entity[3]), entity[4]
			if name != text[start:end]:
				print('wrong match')

	return id2input

def load_pickle():
	with open('data', 'rb') as f:
		content = pickle.load(f)
	return content