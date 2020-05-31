from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import numpy as np
# import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from preprocess import print2logfile

stats_head = '{0:>5}|{1:>7}|{2:>7}|{3:>7}|{4:>7}|{5:>7}|{6:>7}|{7:>7}|{8:>7}|{9:>7}|{10:>7}'
stats_values = '{0:>5}|{1:>6.5f}|{2:>6.5f}|{3:>6.5f}|{4:>6.5f}|{5:>6.5f}|{6:>6.5f}|' + \
               '{7:>6.5f}|{8:>6.5f}|{9:>6.5f}|{10:>6.5f}'

#
#
#
#
#


def run(model, train_data, dev_data, test_data, optimizer, args):

    train_iter = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
    dev_iter = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.batch_size)
    test_iter = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)

    torch.cuda.empty_cache()

    logging.info("Number of training samples {train}, number of dev samples {dev}, number of test samples {test}"
                 .format(train=len(train_data), dev=len(dev_data), test=len(test_data)))

    print2logfile("\n\n################################\nTRAINING STARTED WITH PARAMS: lr=" + str(args.lr), args)
    train(train_iter, dev_iter, test_iter, model, optimizer, args)


#
#
#
#
#


def train(train_iter, dev_iter, test_iter, model, optimizer, args):
    best_dev_f1 = -1

    n_total_steps = len(train_iter)
    total_iter = len(train_iter) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

    logging.info(stats_head.format('Epoch', 'T-Acc', 'T-F1', 'T-Rec', 'T-Prec', 'T-Loss',
                                   'D-Acc', 'D-F1', 'D-Rec', 'D-Prec', 'D-Loss'))


    for epoch in range(args.epochs):

        print2logfile("-------------------------epoch "+ str(epoch) +"-------------------------",args)
        model.train()

        train_loss = 0
        preds = []
        trues = []

        for batch_ids in train_iter:

            input_ids = batch_ids[0].to(args.device)
            att_masks = batch_ids[1].to(args.device)
            labels = batch_ids[2].to(args.device)

            model.zero_grad()

            # forward pass
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)

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

        print2logfile("--Training--", args)
        train_acc, train_f1, train_recall, train_prec = calculate_metrics(trues, preds, args)

        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, args)

        print2logfile("--Validation--", args)
        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred, args)

        logging.info(
            stats_values.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                dev_acc, dev_f1, dev_recall, dev_prec, dev_loss))

        print2logfile(stats_head.format('Epoch', 'T-Acc', 'T-F1', 'T-Rec', 'T-Prec', 'T-Loss',
                                   'D-Acc', 'D-F1', 'D-Rec', 'D-Prec', 'D-Loss'), args)
        print2logfile(stats_values.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                dev_acc, dev_f1, dev_recall, dev_prec, dev_loss), args)

        if best_dev_f1 < dev_f1:
            logging.info('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(
                         dev_acc=dev_f1, best_dev_acc=best_dev_f1))
            print2logfile('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(
                         dev_acc=dev_f1, best_dev_acc=best_dev_f1), args)

            best_dev_f1 = dev_f1
            model_name = 'epoch_{epoch}_dev_f1_{dev_f1:03}.pth.tar'.format(epoch=epoch, dev_f1=dev_f1)
            save_model(model, optimizer, epoch, model_name, args.checkpoint_dir)


        _test_label, _test_pred, test_loss = test(test_iter, model, args)

        print2logfile("\n--Test--", args)
        test_acc, test_f1, test_recall, test_prec = calculate_metrics(_test_label, _test_pred, args)
        logging.info("TEST RESULTS:\nAccuracy: {acc}\nF1: {f1}\nRecall: {recall}\nPrecision: {prec}".format(
                 acc=test_acc, f1=test_f1, recall=test_recall, prec=test_prec))
        print2logfile("TEST RESULTS:\nAccuracy: {acc}\nF1: {f1}\nRecall: {recall}\nPrecision: {prec}".format(
                 acc=test_acc, f1=test_f1, recall=test_recall, prec=test_prec), args)


#
#
#
#
#


def eval(dev_iter, model, args):
    n_total_steps = len(dev_iter)
    model.eval()
    dev_loss = 0
    preds = []
    trues = []
    for batch_ids in dev_iter:
        input_ids = batch_ids[0].to(args.device)
        att_masks = batch_ids[1].to(args.device)
        labels = batch_ids[2].to(args.device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    dev_loss = dev_loss / n_total_steps
    return trues, preds, dev_loss


#
#
#
#
#

def test(test_iter, model, args):
    n_total_steps = len(test_iter)
    model.eval()
    test_loss = 0
    preds = []
    trues = []
    for batch_ids in test_iter:
        input_ids = batch_ids[0].to(args.device)
        att_masks = batch_ids[1].to(args.device)
        labels = batch_ids[2].to(args.device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)
        test_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    test_loss = test_loss / n_total_steps
    return trues, preds, test_loss


#
#
#
#
#


def calculate_metrics(label, pred, args):
    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
    label_class = np.concatenate([numarray for numarray in label]).ravel()

    logging.info('Expected:  {}'.format(label_class[:20]))
    logging.info('Predicted: {}'.format(pred_class[:20]))
    logging.info(classification_report(label_class, pred_class))
    print2logfile('Expected:  {}'.format(label_class[:20]), args)
    print2logfile('Predicted: {}'.format(pred_class[:20]), args)
    print2logfile(classification_report(label_class, pred_class), args)
    acc = accuracy_score(label_class, pred_class)
    f1 = f1_score(label_class, pred_class, average='weighted')
    recall = recall_score(label_class, pred_class, average='weighted')
    prec = precision_score(label_class, pred_class, average='weighted')

    return acc, f1, recall, prec

#
#
#
#
#


def save_model(model, optimizer, epoch, model_name, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_path = os.path.join(checkpoint_dir, model_name)

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)

    logging.info('Best model is saved to {save_path}'.format(save_path=save_path))

    return save_path

#
#
#
#
#


def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint from path "{}" (at epoch {})'.format(
                 checkpoint_path, checkpoint['epoch']))
