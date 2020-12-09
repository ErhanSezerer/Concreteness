import os
import logging
import coloredlogs
import torch
from preprocess import read_data, split_data, read_ground_truth_files, tokenize_data, read_wikidata
from bert import train_bert, test_bert, load_model
from transformers import BertForSequenceClassification, AdamW
import csv
import numpy as np
import random
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from args import args



# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)






def run_bert(device, results_file):
	#prepare the dataset
	logging.info("READING AND PARSING THE DATA...........")
	ground_truth = read_ground_truth_files(args.data_dir)
	if args.mode == "concreteness":
		data_path = os.path.join(args.data_dir, "absconc_data_raw.csv")
		data, labels = read_data(data_path, ground_truth, args.feature)
		train_data, dev_data, test_data, train_labels, dev_labels, test_labels = split_data(data, labels, dev_size=0.1, test_size=0.2)
	elif args.mode == "wiki":
		wiki_data_path = os.path.join(args.data_dir, "articles.csv")
		data, labels = read_wikidata(wiki_data_path, ground_truth)
		train_data, dev_data, test_data, train_labels, dev_labels, test_labels = split_data(data, labels, dev_size=0.1, test_size=0.2)
	train_data, dev_data, test_data = tokenize_data(args, train_data, train_labels, dev_data, dev_labels, test_data, test_labels)

	#prepare the models
	if args.classifier == 'bert':
		model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_label,
                                                          output_attentions=False, output_hidden_states=False)
	elif args.classifier == 'distilbert':
		model = BertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=args.num_label,
                                                          output_attentions=False, output_hidden_states=False)
	optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
	epoch = args.epochs

	train_iter = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
	dev_iter = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.batch_size)
	test_iter = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)

	#create model save directory
	checkpoint_dir = os.path.join(args.checkpoint_dir, args.model_name)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	#run the tests
	logging.info(
        "Number of training samples {train}, number of dev samples {dev}, number of test samples {test}".format(
            train=len(train_data),
            dev=len(dev_data),
            test=len(test_data)))
	train_bert(epoch, model, train_iter, dev_iter, optimizer, device, checkpoint_dir, results_file)

	model = load_model(checkpoint_dir)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_bert(test_iter, model, device)
	del model
	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave










if __name__ == "__main__":
	#initialize
	set_seed(args.seed)
	torch.cuda.empty_cache()
	device = 'cuda' if args.use_gpu and torch.cuda.is_available else 'cpu'


	logger.info('===========Training============')

	#resolve file names and directories
	if args.classifier == "bert":
		file_name = '{model_name}_epochs_{epoch}_lr_{lr}.csv'.format(model_name='bert', epoch=args.epochs, lr=args.lr)
		args.model_name = '{model_name}_epochs_{epoch}_lr_{lr}'.format(model_name='bert', epoch=args.epochs, lr=args.lr)
	elif args.classifier == "distilbert":
		file_name = '{model_name}_epochs_{epoch}_lr_{lr}.csv'.format(model_name='distilbert', epoch=args.epochs, lr=args.lr)
		args.model_name = '{model_name}_epochs_{epoch}_lr_{lr}'.format(model_name='distilbert', epoch=args.epochs, lr=args.lr)

	results_file = os.path.join(args.checkpoint_dir, file_name)
	with open(results_file, 'w') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['Epoch', 'Acc', 'Precision', 'Recall', 'F1', 
			'F1-abs', 'F1-conc', 'P-abs', 'P-conc', 'R-abs', 'R-conc'])

	#run tests
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_bert(device, results_file)

	#print and log the results
	stats_template = '\nAccuracy: {acc}\n' \
                 'F1: {f1}\n' \
                 'ave. F1: {f1_ave}\n' \
                 'Recall: {recall}\n' \
                 'ave. Recall: {recall_ave}\n' \
                 'Precision: {prec}\n' \
                 'ave. Precision: {prec_ave}\n'
	logger.info(stats_template.format(acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave))

	#write results into csv
	with open(results_file, 'a') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['test',
				'%0.4f' % acc,
				'%0.4f' % prec_ave,
				'%0.4f' % recall_ave,
				'%0.4f' % f1_ave,
				'%0.4f' % f1[0],
				'%0.4f' % f1[1],
				'%0.4f' % prec[0],
				'%0.4f' % prec[1],
				'%0.4f' % recall[0],
				'%0.4f' % recall[1]])





