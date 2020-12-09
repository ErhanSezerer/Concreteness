import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from transformers import get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
from args import args


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
logger = logging.getLogger('bert.py')






def train_bert(num_epochs, model, train_iter, dev_iter, optimizer, device, checkpoint_dir, results_file):
    device = torch.device(device)
    best_dev_acc = 0
    best_eval_loss = np.inf

    n_total_steps = len(train_iter)
    total_iter = len(train_iter) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

    for epoch in range(num_epochs):
        model.to(device)
        model.train()

        logger.info('Training epoch: {}'.format(epoch))
        train_loss = 0
        preds = []
        trues = []

        for batch_ids in tqdm(train_iter):
            input_ids = batch_ids[0].to(device)
            att_masks = batch_ids[1].to(device)
            labels = batch_ids[2].to(device)

            model.zero_grad()

            # forward pass
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

            # record preds, trues
            _pred = logits.cpu().data.numpy()
            preds.append(_pred)
            _label = labels.cpu().data.numpy()
            trues.append(_label)

            train_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec, train_f1_ave, train_recall_ave, train_prec_ave = calculate_metrics(trues, preds, average=None)
        print(stats_template.format(mode='train', epoch_idx=epoch, acc=train_acc, f1=train_f1, f1_ave=train_f1_ave, recall=train_recall,
                  recall_ave=train_recall_ave, prec=train_prec, prec_ave=train_prec_ave, loss=train_loss))


        #validation
        acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, valid_loss = eval_bert(dev_iter, model, device)


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







def eval_bert(dev_iter, model, device):
    device = torch.device(device)
    n_total_steps = len(dev_iter)
    model.to(device)
    model.eval()
    dev_loss = 0
    preds = []
    trues = []
    for batch_ids in tqdm(dev_iter):
        input_ids = batch_ids[0].to(device)
        att_masks = batch_ids[1].to(device)
        labels = batch_ids[2].to(device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    dev_loss = dev_loss / n_total_steps
    acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(trues, preds, average=None)
    print(stats_template
          .format(mode='valid', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss=dev_loss))
    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave, dev_loss







def test_bert(test_iter, model, device):
    device = torch.device(device)
    model.to(device)
    model.eval()
    trues = []
    preds = []
    for batch_ids in tqdm(test_iter):
        input_ids = batch_ids[0].to(device)
        att_masks = batch_ids[1].to(device)
        labels = batch_ids[2].to(device)

         # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

         # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    if args.print_preds == True:
        pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in preds]).ravel()
        path = os.path.join(args.checkpoint_dir, "preds_bert.csv")
        with open(path, 'w') as output_file: 
            cw = csv.writer(output_file, delimiter='\t')
            for pred in pred_class:
                cw.writerow(str(pred))

    acc, f1, recall, prec, f1_ave, recall_ave, prec_ave = calculate_metrics(trues, preds, average=None)
    print(stats_template
          .format(mode='test', epoch_idx='__', acc=acc, f1=f1, f1_ave=f1_ave, recall=recall,
                  recall_ave=recall_ave, prec=prec, prec_ave=prec_ave, loss='__'))
    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave








def calculate_metrics(label, pred, average='binary'):
    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
    label_class = np.concatenate([numarray for numarray in label]).ravel()

    logging.debug('Expected: \n{}'.format(label_class[:20]))
    logging.debug('Predicted: \n{}'.format(pred_class[:20]))

    acc = round(accuracy_score(label_class, pred_class), 4)
    f1 = [round(score, 4) for score in f1_score(label_class, pred_class, average=average)]
    recall = [round(score, 4) for score in recall_score(label_class, pred_class, average=average)]
    prec = [round(score, 4) for score in precision_score(label_class, pred_class, average=average)]

    f1_ave = f1_score(label_class, pred_class, average='weighted')
    recall_ave = recall_score(label_class, pred_class, average='weighted')
    prec_ave = precision_score(label_class, pred_class, average='weighted')

    return acc, f1, recall, prec, f1_ave, recall_ave, prec_ave




def save_model(model, checkpoint_dir):
    model.save_pretrained(checkpoint_dir)
    return


def load_model(checkpoint_path):
    model = BertForSequenceClassification.from_pretrained(checkpoint_path, num_labels=args.num_label,
                                                          output_attentions=False, output_hidden_states=False)
    return model


