import random
import re
import pandas as pd
from nltk.corpus import stopwords
import csv
import os
import logging
from torch.utils.data import TensorDataset
import torch






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








def read_data(path, ground_truth_scores):
	field = 'caption'


	#read the dataset
	df_data = pd.read_csv(path, delimiter='\t')
	df_field = df_data[['word',field]].dropna()
	print("dataset size:" + str(len(df_data)))
	print("field's non-empty size:" + str(len(df_field)))


	#filter out stop words
	stop_words = set(stopwords.words('english'))
	df_field_filtered = df_field[~df_field['word'].isin(stop_words)]
	

	#extract caption strings
	df_grouped_filtered = df_field_filtered.groupby('word')[field].apply(list).reset_index(name='captions')
	raw_data = df_grouped_filtered.to_numpy()

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

	if args.model == 'bert':
		# tokenize the text with BERT ids
		logging.info("Loading BERT tokenizer...")
		tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
	elif args.model == 'distilbert':
		# tokenize the text with DistilBERT ids
		logging.info("Loading DistilBERT tokenizer...")
		tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

	logging.info("Tokenizing train set which has {} answers...".format(len(train_articles)))
	train_ids, train_att_mask = tokenize_helper(args, train_articles, tokenizer)
	logging.info("Tokenizing dev set which has {} answers...".format(len(dev_articles)))
	dev_ids, dev_att_mask = tokenize_helper(args, dev_articles, tokenizer)
	logging.info("Tokenizing test set which has {} answers...".format(len(test_articles)))
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





def print2logfile(string, args):
	filename = args.model + "_logs.txt"
	log_path = os.path.join(args.checkpoint_dir, filename)
	with open(log_path, "a") as logfile:
		logfile.write(string + "\n")

