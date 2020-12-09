import random
import re
import pandas as pd
from nltk.corpus import stopwords
import csv
import os
import logging
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
from transformers import AutoTokenizer
import numpy as np
from args import args
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision import models



def split_data(data, labels, dev_size=0.1, test_size=0.2):
	test_count = int(len(data)*test_size)
	dev_count = int(len(data)*dev_size)

	test = data[ : test_count]
	dev = data[test_count : test_count+dev_count]
	train = data[test_count+dev_count : ]

	test_labels = labels[ : test_count]
	dev_labels = labels[test_count : test_count+dev_count]
	train_labels = labels[test_count+dev_count : ]

	return train, dev, test, train_labels, dev_labels, test_labels








def read_data(path, ground_truth_scores, field):

	#read the dataset
	df_data = pd.read_csv(path, delimiter='\t')
	df = df_data[df_data.index.isin([1,3])]
	print(df)
	exit()
	df_field = df_data[['word',field]].dropna()
	logging.info("dataset size:" + str(len(df_data)))
	logging.info("field's non-empty size:" + str(len(df_field)))

	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_field_filtered = df_field[~df_field['word'].isin(stop_words)]

	#extract caption strings
	#df_grouped_filtered = df_field_filtered.groupby('word')[field].apply(list).reset_index(name='captions')
	raw_data = df_field_filtered.to_numpy()

	data = []
	label = []
	for cap in raw_data:
		caption = ""
		for sentence in cap[1]:
			caption += "." + sentence
		data.append(caption)
		if ground_truth_scores[cap[0]] >= 400:
			label.append(1)
		else:
			label.append(0)
	
	
	#shuffle the data
	temp = list(zip(data, label))
	random.shuffle(temp)
	data, label = zip(*temp)

	return data, label




def read_dataWithIndices(path, ground_truth_scores, field, indices):

	#read the dataset
	df_data = pd.read_csv(path, delimiter='\t')
	df_data = df_data[df_data.index.isin(indices)]
	df_field = df_data[['word',field]].dropna()
	logging.info("dataset size:" + str(len(df_data)))
	logging.info("field's non-empty size:" + str(len(df_field)))

	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_field_filtered = df_field[~df_field['word'].isin(stop_words)]

	#extract caption strings
	#df_grouped_filtered = df_field_filtered.groupby('word')[field].apply(list).reset_index(name='captions')
	raw_data = df_field_filtered.to_numpy()

	data = []
	label = []
	for cap in raw_data:
		caption = ""
		for sentence in cap[1]:
			caption += "." + sentence
		data.append(caption)
		if ground_truth_scores[cap[0]] >= 400:
			label.append(1)
		else:
			label.append(0)

	return data, label



def find_bertdata(data_iter, field):
	data_path = os.path.join(args.data_dir, "absconc_data_raw.csv")
	ground_truth_scores = read_ground_truth_files(args.data_dir)

	pathlist = []
	for inputs, labels, paths in tqdm(data_iter):
		for path in paths:
			pathlist.append(int(path.split('/')[-1].split('.')[0]))


	#read the dataset
	df_data = pd.read_csv(data_path, delimiter='\t')
	df_data = df_data[df_data.index.isin(pathlist)]
	df_indexed = df_data.reindex(pathlist)

	df_field = df_indexed[['word',field]].dropna()
	logging.info("dataset size:" + str(len(df_data)))
	logging.info("field's non-empty size:" + str(len(df_field)))

	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_field_filtered = df_field[~df_field['word'].isin(stop_words)]

	#extract caption strings
	#df_grouped_filtered = df_field_filtered.groupby('word')[field].apply(list).reset_index(name='captions')
	raw_data = df_field_filtered.to_numpy()

	data = []
	label = []
	for cap in raw_data:
		caption = ""
		#for sentence in cap[1]:
			#caption += "." + sentence
		print(caption)
		data.append(caption)
		if ground_truth_scores[cap[0]] >= 400:
			label.append(1)
		else:
			label.append(0)

	train_data = tokenize_dataWithIndices(data, label)
	train_iter = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=args.batch_size)
	return train_iter




def read_wikidata(path, ground_truth_scores):
	words = ground_truth_scores.keys()

	#read the dataset
	iter_csv = pd.read_csv(path, delimiter='\t', iterator=True, chunksize=200000)
	df_data = pd.concat([chunk[chunk['Title'].str.lower().isin(words)] for chunk in iter_csv])
	df_data['Title'] = df_data['Title'].str.lower()	

	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_data_filtered = df_data[~df_data['Title'].isin(stop_words)]
	df_data_filtered = df_data_filtered.dropna()
	np_data = df_data_filtered.to_numpy()

	data = []
	label = []
	for row in np_data:
		article_paragraphs = row[1].split('\n')
		text = clean_wikitext(article_paragraphs[0])
		data.append(text)

		if ground_truth_scores[row[0]] >= 400:
			label.append(1)
		else:
			label.append(0)


	#shuffle the data
	temp = list(zip(data, label))
	random.shuffle(temp)
	data, label = zip(*temp)

	return data, label

		
	


def clean_wikitext(text):
	partial_text = re.split(r'\[\[|]]',text)
	clean_text = ""
	for i in range(len(partial_text)):
		if i%2 == 0:
			clean_text += partial_text[i]
		else:	
			if '|' in partial_text[i]:
				clean_text += partial_text[i].split('|')[1]
			else:
				clean_text += partial_text[i]
	return clean_text	








def read_ground_truth_files(path):
	ground_truth = {}

	abs_path = os.path.join(path, "100-400.txt")
	with open(abs_path, 'r') as abstracts:
		for line in abstracts:
			word = re.search("^[a-zA-Z]+", line).group(0).lower()
			score = int(re.search("\d+", line).group(0))
			ground_truth[word] = score
	conc_path = os.path.join(path, "400-700.txt")
	with open(conc_path, 'r') as concretes:
		for line in concretes:
			word = re.search("^[a-zA-Z]+", line).group(0).lower()
			score = int(re.search("\d+", line).group(0))
			ground_truth[word] = score
	return ground_truth






########################################################################################################
#FOR BERT#
########################################################################################################
def tokenize_data(args, train_articles, train_labels, dev_articles, dev_labels, test_articles, test_labels):

	if args.classifier == 'bert':
		# tokenize the text with BERT ids
		logging.info("Loading BERT tokenizer...")
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	elif args.classifier == 'distilbert':
		# tokenize the text with DistilBERT ids
		logging.info("Loading DistilBERT tokenizer...")
		tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

	logging.info("Tokenizing train set which has {} samples...".format(len(train_articles)))
	train_ids, train_att_mask = tokenize_helper(args, train_articles, tokenizer)
	logging.info("Tokenizing dev set which has {} samples...".format(len(dev_articles)))
	dev_ids, dev_att_mask = tokenize_helper(args, dev_articles, tokenizer)
	logging.info("Tokenizing test set which has {} samples...".format(len(test_articles)))
	test_ids, test_att_mask = tokenize_helper(args, test_articles, tokenizer)

	logging.info("Converting train, dev and test sets to torch tensors...")
	train_ids = torch.cat(train_ids, dim=0)
	dev_ids = torch.cat(dev_ids, dim=0)
	test_ids = torch.cat(test_ids, dim=0)
	train_att_mask = torch.cat(train_att_mask, dim=0)
	dev_att_mask = torch.cat(dev_att_mask, dim=0)
	test_att_mask = torch.cat(test_att_mask, dim=0)
	train_labels = torch.tensor(train_labels)
	dev_labels = torch.tensor(dev_labels)
	test_labels = torch.tensor(test_labels)

	train_dataset = TensorDataset(train_ids, train_att_mask, train_labels)
	dev_dataset = TensorDataset(dev_ids, dev_att_mask, dev_labels)
	test_dataset = TensorDataset(test_ids, test_att_mask, test_labels)

	return train_dataset, dev_dataset, test_dataset



def tokenize_dataWithIndices(train_articles, train_labels):

	logging.info("Loading BERT tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

	logging.info("Tokenizing train set which has {} samples...".format(len(train_articles)))
	train_ids, train_att_mask = tokenize_helper(args, train_articles, tokenizer)

	logging.info("Converting train, dev and test sets to torch tensors...")
	train_ids = torch.cat(train_ids, dim=0)
	train_att_mask = torch.cat(train_att_mask, dim=0)
	train_labels = torch.tensor(train_labels)

	train_dataset = TensorDataset(train_ids, train_att_mask, train_labels)

	return train_dataset





def tokenize_helper(args, articles, tokenizer):

	ids = []
	att_mask = []
	for article in articles:
		encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                    pad_to_max_length=True, return_attention_mask=True,
                                                    return_tensors='pt')
		ids.append(encoded_article['input_ids'])
		att_mask.append(encoded_article['attention_mask'])

	return ids, att_mask






########################################################################################################
#FOR RESNET#
########################################################################################################
class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def open_images(ground_truth_scores, field):

	#start getting images
	mean_nums = [0.485, 0.456, 0.406]
	std_nums = [0.229, 0.224, 0.225]
	transforms = {'train': T.Compose([
			T.RandomResizedCrop(size=224),
			T.RandomRotation(degrees=15),
			T.RandomHorizontalFlip(),
			T.ToTensor(),
			T.Normalize(mean_nums, std_nums)]), 
				'val': T.Compose([
			T.Resize(size=224),
			T.CenterCrop(size=224),
			T.ToTensor(),
			T.Normalize(mean_nums, std_nums)]),
				'test': T.Compose([
			T.Resize(size=224),
			T.CenterCrop(size=224),
			T.ToTensor(),
			T.Normalize(mean_nums, std_nums)])}
	dataset = ImageFolderWithPaths(args.image_path, transform=transforms['train'])#supported files .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp

	return dataset















