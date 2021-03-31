# Antonio Robayo
# amr1059
import os
from copy import copy
import logging
import pandas as pd
import pytz
import argparse
from datetime import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from transformers import AutoTokenizer, AutoModel

TIMEZONE = pytz.timezone('America/New_York')
SCRATCH = os.environ['SCRATCH'] + '/'

def get_data_path(source):
    data_source = {'multiple':'data/multi_emoji/',
                   'single':'data/single_emoji/',
                   'full':'data/full_data/',
                   'no_repeats':'data/no_repeats/'}
    file_names = {'multiple':'_multi_emoji',
                  'single':'_single_emoji',
                  'full':'',
                  'no_repeats':'_no_repeats'}
    train = data_source[source] + 'emoji_nsp_dataset' + file_names[source] + '_train.csv'
    valid = data_source[source] + 'emoji_nsp_dataset' + file_names[source] + '_valid.csv'
    test = data_source[source] + 'emoji_nsp_dataset' + file_names[source] + '_test.csv'
    return train, valid, test

def calculate_metrics(y_truth, y_preds):
    f1 = f1_score(y_truth, y_preds)
    accuracy = accuracy_score(y_truth, y_preds)
    return f1, accuracy

def show_quantiles(df, column):
    percentiles = [0.5, 0.75, 0.95, 1]
    print('Percentiles for: {}'.format(column))
    for i in percentiles:
        print('{}th percentile: {}'.format(i*100 ,df[column].map(lambda x: len(x)).quantile(q = i)))

def tokenize_data(df, tokenizer, max_sentence_length=225):
    df_ = df.copy()
    df_.dropna(inplace = True)
    tweets = df_['tweets'] + ' ' + tokenizer.sep_token + ' ' + df_['emoji_sentence']
    tokenized_tweets = []
    attn_masks = []
    for idx, tweet in tweets.items():
        encoded = tokenizer.encode_plus(tweet, padding='max_length',
                                        truncation=True, max_length=max_sentence_length)
        tokenized_tweets.append(encoded['input_ids'])
        attn_masks.append(encoded['attention_mask'])
    tokenized_tweets = torch.tensor(tokenized_tweets, dtype=torch.long)
    attn_masks = torch.tensor(attn_masks, dtype=torch.long)
    labels = torch.tensor(df_['follows?'].values.tolist(), dtype=torch.long)
    return TensorDataset(tokenized_tweets, attn_masks, labels)

def get_loader(dataset, batch_size):
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset,sampler=sampler, batch_size=batch_size)
    return loader

class bertmoji(nn.Module):
    def __init__(self, roberta, num_classes=1):
        super().__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(roberta.config.hidden_dropout_prob)
        self.W = nn.Linear(roberta.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        h_cls = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[1]
        # h_cls = h_cls_pool[:, 0]
        logits = self.dropout(self.W(h_cls))
        return logits

class trainer(object):
    def __init__(self, model, device, criterion, train_data, val_data, test_data):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
    def train_step(self, batch_size, optimizer):
        self.model.to(self.device)
        self.model.train()
        train_loader = get_loader(self.train_data, batch_size)
        train_loss_cache = []
        y_preds = []
        y_truth = []
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_masks, labels = batch
            logits = self.model(input_ids, attention_masks)
            logits = logits.to(self.device)
            labels = labels.type_as(logits)
            loss = self.criterion(logits.squeeze(-1), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.round(torch.sigmoid(logits))
            preds = preds.view_as(labels)
            acc  = preds.eq(labels).sum().item() / labels.size(0)
            y_preds += preds.tolist()
            y_truth += labels.tolist()
            train_loss_cache.append(loss.item())
            current_avg_loss = np.sum(train_loss_cache) / (step + 1)
            if step % 100 == 0:
                print('{} | Avg. BCE loss: {}, Accuracy: {} | {}/{}'.format(dt.now(tz=TIMEZONE), current_avg_loss, 
                                                                            acc, step, len(train_loader)))
        evaluated_loss = np.mean(train_loss_cache)
        f1, accuracy = calculate_metrics(y_truth, y_preds)
        return evaluated_loss, f1, accuracy
    def evaluate_step(self, batch_size, use_test=False):
        self.model.to(self.device)
        self.model.eval()
        if use_test:
            loader = get_loader(self.test_data, batch_size)
        else:
            loader = get_loader(self.val_data, batch_size)
        eval_loss_cache = []
        y_preds = []
        y_truth = []
        for step, batch in enumerate(loader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, attention_masks, labels = batch
            logits = self.model(input_ids, attention_masks)
            logits = logits.to(self.device)
            labels = labels.type_as(logits)
            loss = self.criterion(logits.squeeze(-1), labels)
            
            preds = torch.round(torch.sigmoid(logits))
            preds = preds.view_as(labels)
            y_preds += preds.tolist()
            y_truth += labels.tolist()
            eval_loss_cache.append(loss.item())
        evaluated_loss = np.sum(eval_loss_cache) / len(loader)
        f1, accuracy = calculate_metrics(y_truth, y_preds)
        return evaluated_loss, f1, accuracy

    def train(self, batch_size, epochs, optimizer, patience, data_source, save_path='./'):
        best_val_loss = np.inf
        patience_counter = 0
        train_performance = {'loss':[], 'f1':[], 'accuracy':[]}
        valid_performance = {'loss':[], 'f1':[], 'accuracy':[]}
        for epoch in range(epochs):
            print('{} | Epoch {}'.format(dt.now(tz=TIMEZONE), epoch + 1))
            loss, f1, accuracy = self.train_step(batch_size, optimizer)
            train_performance['loss'].append(loss)
            train_performance['f1'].append(f1)
            train_performance['accuracy'].append(accuracy)
            previous_val_loss = copy(best_val_loss)
            avg_val_loss, val_f1, val_accuracy = self.evaluate_step(batch_size)
            if avg_val_loss < best_val_loss:
                best_val_loss = copy(avg_val_loss)
            print('{} | Validation loss: {}, F1: {}, Accuracy: {}'.format(dt.now(tz=TIMEZONE), 
                                                                          best_val_loss,
                                                                          val_f1,
                                                                          val_accuracy))
            valid_performance['loss'].append(loss)
            valid_performance['f1'].append(f1)
            valid_performance['accuracy'].append(accuracy)
            if patience_counter > patience:
                print('{} | Stopping early.'.format(dt.now(tz=TIMEZONE)))
                break
            if best_val_loss < previous_val_loss:
                print('{} | Saving model...'.format(dt.now(tz=TIMEZONE)))
                torch.save({'model': self.model.state_dict(),
                            'train_performance': train_performance,
                            'valid_performance': valid_performance}, save_path + data_source + '-bertmoji.pt')
                patience_counter = 0
            else:
                patience_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run bertmoji model')
    parser.add_argument('--max_sentence_length',
                        type=int,
                        default=128,
                        help='Max sequence length for RoBERTa tokenizing and subsequent encoding.')
    parser.add_argument('--num_epochs',
                            type=int,
                            default=20,
                            help='Number of epochs to train model.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Controls how much we are adjusting the weights of our network with\
                              respect to the loss gradient.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='L2 regularization penalty. Causes weights to exponentially decay\
                              to zero.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Size of batch, i.e. size of data partitions')
    parser.add_argument('--logging_file',
                        type=str,
                        help='Name of experiment to identify logging file.')
    parser.add_argument('--evaluate_only',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Calculate model performance on test set only')
    parser.add_argument('--patience', 
                        type=int,
                        default=5,
                        help='Max patience for whether or not to continue training process.')
    parser.add_argument('--data_source', 
                        type=str,
                        default='full',
                        help='Data to use for training.')
    # TO-DO add argument + logic for loading fine-tuned model
    args = parser.parse_args()

    if args.logging_file:
        log_filename = args.logging_file + '_' + str(dt.now(tz=TIMEZONE).date())
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler(log_filename+'.log', 'a'))
        print = logger.info

    train_file, valid_file, test_file = get_data_path(args.data_source)
    train = pd.read_csv(SCRATCH + train_file)
    valid = pd.read_csv(SCRATCH + valid_file)
    test = pd.read_csv(SCRATCH + test_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL = "cardiffnlp/twitter-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    # make [USER] a token to be learned rather than treating as unknown
    tokenizer.add_tokens(["[USER]"])
    model.resize_token_embeddings(len(tokenizer)) 

    tokenized_test = tokenize_data(test, tokenizer, max_sentence_length=args.max_sentence_length)
    tokenized_valid = tokenize_data(valid, tokenizer, max_sentence_length=args.max_sentence_length)
    tokenized_train = tokenize_data(train, tokenizer, max_sentence_length=args.max_sentence_length)

    bceloss = nn.BCEWithLogitsLoss()
    bertmoji_classifier = bertmoji(model)
    optimizer = optim.Adam(bertmoji_classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bertmoji_trainer = trainer(bertmoji_classifier, device, bceloss, tokenized_train, tokenized_valid, tokenized_test)
    if args.evaluate_only:
        evaluated_loss, f1, accuracy = bertmoji_trainer.evaluate_step(batch_size=args.batch_size, use_test=True)
        print('Avg. BCE loss: {}'.format(evaluated_loss))
        print('F1 score: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))
    else:
        bertmoji_trainer.train(args.batch_size, args.num_epochs, optimizer, args.patience, args.data_source, save_path=SCRATCH)
