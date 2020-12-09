import os
import logging
import coloredlogs
import torch
from preprocess import read_ground_truth_files, open_images
from resnet import train_resnet, test_resnet, load_model
import csv
import numpy as np
import random
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, random_split
from args import args
from torchvision import models
from torch.optim import lr_scheduler
from torch import nn, optim


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




def run_resnet(device, results_file):
	#prepare the dataset
	logging.info("READING AND PARSING THE DATA...........")
	ground_truth = read_ground_truth_files(args.data_dir)
	dataset = open_images(ground_truth, args.feature)
		
	#split the dataset
	size = len(dataset)
	train_size = int(size * 0.7)
	val_size = int(size * 0.1)
	test_size = (size - train_size) - val_size
	train_data, dev_data, test_data = random_split(dataset, [train_size,val_size,test_size])

	train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
	dev_iter = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
	test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

	for images, labels, path in test_iter:
		print(path)
		exit()

	model = models.resnet152(pretrained=True)
	n_features = model.fc.in_features
	model.fc = nn.Linear(n_features, args.num_label)

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	loss_fn = nn.CrossEntropyLoss().to(device)	
	epoch = args.epochs


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
	train_resnet(epoch, model, train_iter, dev_iter, optimizer, loss_fn, scheduler, device, checkpoint_dir, results_file)

	model = load_model(model, checkpoint_dir)
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = test_resnet(test_iter, model, loss_fn, device)
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
	elif args.classifier == "resnet":
		file_name = '{model_name}_epochs_{epoch}_lr_{lr}.csv'.format(model_name='resnet', epoch=args.epochs, lr=args.lr)
		args.model_name = '{model_name}_epochs_{epoch}_lr_{lr}'.format(model_name='resnet', epoch=args.epochs, lr=args.lr)


	results_file = os.path.join(args.checkpoint_dir, file_name)
	with open(results_file, 'w') as output_file:
		cw = csv.writer(output_file, delimiter='\t')
		cw.writerow(['Epoch', 'Acc', 'Precision', 'Recall', 'F1', 
			'F1-abs', 'F1-conc', 'P-abs', 'P-conc', 'R-abs', 'R-conc'])

	#run tests
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = run_resnet(device, results_file)

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





