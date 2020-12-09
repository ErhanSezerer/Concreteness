import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from tqdm import tqdm
from args import args

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'
stats_template = 'Epoch {epoch_idx}\n' \
                 '{mode} Accuracy: {acc}\n' \
                 '{mode} F1: {f1}\n' \
                 '{mode} ave. F1: {f1_ave}\n' \
                 '{mode} Recall: {recall}\n' \
                 '{mode} ave. Recall: {recall_ave}\n' \
                 '{mode} Precision: {prec}\n' \
                 '{mode} ave. Precision: {prec_ave}\n' \
                 '{mode} Loss: {loss}\n'
logger = logging.getLogger('resnet.py')








def train_resnet(num_epochs, model, train_iter, dev_iter, optimizer, loss_fn, scheduler, device, checkpoint_dir, results_file):
	device = torch.device(device)
	best_dev_acc = 0
	best_eval_loss = np.inf

	n_total_steps = len(train_iter)
	total_iter = len(train_iter) * num_epochs


	for epoch in range(num_epochs):
		model.to(device)
		model.train()
		truths = []
		predictions = []

		logger.info('Training epoch: {}'.format(epoch))
		train_loss = 0

		for inputs, labels in tqdm(train_iter):
			inputs = inputs.to(device)
			labels = labels.to(device)

			# forward pass
			outputs = model(inputs)
			_, preds = torch.max(outputs, dim=1)
			loss = loss_fn(outputs, labels)
			train_loss += loss.item()
			# record preds, trues
			predictions.extend(preds)
			truths.extend(labels)

            # backpropagate and update optimizer learning rate
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		scheduler.step()

		predictions = torch.as_tensor(predictions).cpu()
		truths = torch.as_tensor(truths).cpu()
		train_loss = train_loss / n_total_steps

		train_acc, train_f1, train_recall, train_prec, train_f1_ave, train_recall_ave, train_prec_ave = calculate_metrics(truths, predictions, average=None)
		print(stats_template.format(mode='train', epoch_idx=epoch, acc=train_acc, f1=train_f1, f1_ave=train_f1_ave, recall=train_recall,
                  recall_ave=train_recall_ave, prec=train_prec, prec_ave=train_prec_ave, loss=train_loss))

        #validation
		acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, valid_loss = eval_resnet(dev_iter, model, loss_fn, device)


		#write results to csv
		with open(results_file, 'a') as output_file:
			cw = csv.writer(output_file, delimiter='\t')
			cw.writerow(["train-"+str(epoch),
				'%0.4f' % train_acc,
				'%0.4f' % train_prec_ave,
				'%0.4f' % train_recall_ave,
				'%0.4f' % train_f1_ave,
				'%0.4f' % train_f1[0],
				'%0.4f' % train_f1[1],
				'%0.4f' % train_prec[0],
				'%0.4f' % train_prec[1],
				'%0.4f' % train_recall[0],
				'%0.4f' % train_recall[1]])
			cw.writerow(["valid-"+str(epoch),
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

        #save_model(model, checkpoint_dir)
		if best_dev_acc < f1_ave:
			logging.debug('New dev f1 {dev_acc} is larger than best dev f1 {best_dev_acc}'.format(dev_acc=f1, best_dev_acc=best_dev_acc))
			best_dev_acc = f1_ave
			best_eval_loss = valid_loss
			save_model(model, checkpoint_dir)







def eval_resnet(dev_iter, model, loss_fn, device):
	device = torch.device(device)
	n_total_steps = len(dev_iter)
	model.to(device)
	model.eval()
	dev_loss = 0
	predictions = []
	truths = []

	# forward pass
	with torch.no_grad():
		for inputs, labels in tqdm(dev_iter):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, dim=1)
			loss = loss_fn(outputs, labels)
			dev_loss += loss.item()

			# record preds, trues
			predictions.extend(preds)
			truths.extend(labels)

	predictions = torch.as_tensor(predictions).cpu()
	truths = torch.as_tensor(truths).cpu()

	dev_loss = dev_loss / n_total_steps
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions, average=None)
	print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=dev_loss))
	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, dev_loss







def test_resnet(test_iter, model, loss_fn, device):
	device = torch.device(device)
	n_total_steps = len(test_iter)
	model.to(device)
	model.eval()
	dev_loss = 0
	predictions = []
	truths = []

	# forward pass
	with torch.no_grad():
		for inputs, labels in tqdm(test_iter):
			inputs = inputs.to(device)
			labels = labels.to(device)

			outputs = model(inputs)
			_, preds = torch.max(outputs, dim=1)
			loss = loss_fn(outputs, labels)
			dev_loss += loss.item()

			# record preds, trues
			predictions.extend(preds)
			truths.extend(labels)

	predictions = torch.as_tensor(predictions).cpu()
	truths = torch.as_tensor(truths).cpu()

	dev_loss = dev_loss / n_total_steps
	acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(truths, predictions, average=None)
	print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=dev_loss))
	return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave








def calculate_metrics(label, pred, average='binary'):
    logging.debug('Expected: \n{}'.format(label[:20]))
    logging.debug('Predicted: \n{}'.format(pred[:20]))

    acc = round(accuracy_score(label, pred), 4)
    f1 = [round(score, 4) for score in f1_score(label, pred, average=average)]
    recall = [round(score, 4) for score in recall_score(label, pred, average=average)]
    prec = [round(score, 4) for score in precision_score(label, pred, average=average)]

    f1_ave = f1_score(label, pred, average='weighted')
    recall_ave = recall_score(label, pred, average='weighted')
    prec_ave = precision_score(label, pred, average='weighted')

    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave




def save_model(model, checkpoint_dir):
	checkpoint_dir = os.path.join(checkpoint_dir, '{model_name}_epochs_{epoch}_lr_{lr}.bin'.format(model_name='resnet', epoch=args.epochs, lr=args.lr))
	torch.save(model.state_dict(), checkpoint_dir)
	return


def load_model(model, checkpoint_dir):
	checkpoint_dir = os.path.join(checkpoint_dir, '{model_name}_epochs_{epoch}_lr_{lr}.bin'.format(model_name='resnet', epoch=args.epochs, lr=args.lr))
	model.load_state_dict(torch.load(checkpoint_dir))
	return model


