import os
import logging
import argparse
import coloredlogs
import torch
from str2bool import str2bool
from parameters import PARAM
from preprocess import read_data, split_data, read_ground_truth_files, tokenize_data, print2logfile
from bert import run
from transformers import AutoConfig, AutoModelForSequenceClassification, AdamW

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)
default_random_seed = 42


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True








def main():
	parser = argparse.ArgumentParser(description='Concreteness Tests')
	parser.add_argument('--seed', default=default_random_seed, type=int)
	parser.add_argument('--batch_size', default=8, type=int)
	parser.add_argument('--epochs', default=5, type=int)
	parser.add_argument('--lr', default=1e-5, type=float)
	parser.add_argument('--device', default='cuda', type=str, help="(options ['cpu', 'cuda'] defaults to 'cpu')")
	parser.add_argument('--checkpoint_dir', default='./models', type=str)
	parser.add_argument('--MAX_LEN', default=512, type=int)
	parser.add_argument('--model', default='bert', type=str, help="(options ['bert', 'distilbert'] defaults to 'bert')")
	parser.add_argument('--num_labels', default=2, type=int, help="Number of classes in dataset")

	args = parser.parse_args()

	print("READING AND PARSING THE DATA...........")
	data_path = os.path.join(PARAM.root_filepath, "absconc_data_raw.csv")
	ground_truth = read_ground_truth_files(PARAM.ground_truth_path)
	data, labels = read_data(data_path, ground_truth)
	train_data, dev_data, test_data, train_labels, dev_labels, test_labels = split_data(data, labels, dev_size=0, test_size=0.2)
	print("Dataset: " +str(len(train_data))+ " train, " +str(len(dev_data))+ " dev, " +str(len(test_data))+ " test")
	

	

	train_data, dev_data, test_data = tokenize_data(args, train_data, train_labels, dev_data, dev_labels, test_data, test_labels)

	if args.model == 'bert':
		logging.info('Starting executions with BERT...')
		config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=args.num_labels)
		model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
	elif args.model == 'distilbert':
		logging.info('Starting executions with DistilBERT...')
		config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=args.num_labels)
		model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

	if torch.cuda.is_available():
		args.device = torch.device('cuda')
		model.cuda()
	else:
		args.device = torch.device('cpu')
		model.cpu()

	optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

	run(model, train_data, dev_data, test_data, optimizer, args)





if __name__ == "__main__":
	main()
