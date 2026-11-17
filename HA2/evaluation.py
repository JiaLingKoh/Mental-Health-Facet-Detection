# ## Testing
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
#from transformers import BertConfig,BertModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
    
from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
import os
import csv
import datetime
from argparse import ArgumentParser

# import custom library
from lib.utils import *
import importlib
import sys
importlib.reload(sys.modules['lib.utils'])

model_thresholds = {'-2':0.9,'-1':0.9,'1':0.9,'2':0.9,'0':0.9,'5':0.9, '-3':0.9}
np_model = '5_model'

# import lib library
from lib.model import *
importlib.reload(sys.modules['lib.model'])

torch.manual_seed(0)

now = datetime.datetime.now()
start_timestamp = now.strftime('%Y-%m-%d %H:%M')

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = model.to(device)

round_times = 5

def testing(n, round_times, label_list, label_dict, path):
# for n in range(round_times):
    pred_list = []
    prob_list = []
    score_list = []
    
    round_path = 'round_{}'.format(n)

    for l in label_list:
        i = label_dict[l]
        print("Run model: ", i)
        df_test = pd.read_csv('data/data_4test{}/{}/test{}.csv'.format(NP, round_path,i))
        df_test_orig = df_test
        df_test = df_test.loc[:, ['sentence','label']]
        testset = SentenceDataset("test", tokenizer=tokenizer, df=df_test)
        testloader = DataLoader(testset, batch_size= 32, 
                                collate_fn=create_mini_batch)
        # testing mode
        model.eval()

        # load saved models
        model.load_state_dict(torch.load('model_f1/{}/{}.ckpt'.format(round_path,i)))
        
        predictions, probs, test_acc, _ , _ = get_predictions(model, testloader, act_fun = 'softmax', use_threshold = True, threshold = model_thresholds[l])

        predictions = predictions.cpu()

        df = pd.DataFrame({"predict": predictions})
        test_label_list = df_test.loc[:, 'label'].to_list()
        df_label_test = list(map(trans,test_label_list))

        precision,recall,f1,_ = metrics.precision_recall_fscore_support(df_label_test,predictions)
        accuracy = metrics.accuracy_score(df_label_test,predictions)

        scores = [accuracy,precision,recall,f1]
        score_list.append(scores)
#         save_scores_table(scores,'scores_{}.csv'.format(i))

        index_map = {v: k for k, v in testset.label_map.items()}

        pred_list.append(df['predict'])
        prob_list.append(probs)

        df['predict'] = df.predict.apply(lambda x: index_map[x])

        df_pred = pd.concat([df_test_orig.loc[:, ['id','sentence','label']], 
                  df.loc[:, 'predict']], axis=1)
        #df_pred = df_pred.sort_values(by='id')
        
        if path:
           result_path = 'result/'+path+'/{}'.format(round_path)
        else:
           result_path = 'result/{}'.format(round_path)   
        os.system('mkdir -p '+result_path)
        df_pred.to_csv('result/{}/precdictions{}.csv'.format(round_path,i),encoding='utf-8_sig', index=False)
    
    #save_scores_table(score_list,'result/{}/scores_all.csv'.format(round_path),multi_model=True)


    # output prediction_all file
    df_test_all = pd.read_csv('data/data_4test{}/{}/test.csv'.format(NP, round_path))

    
    for i,pred in enumerate(pred_list):
        pred = pred.to_frame(name='model_for_{}'.format(label_list[i]))
        prob = pd.DataFrame({'prob_{}'.format(label_list[i]): list(prob_list[i].cpu())})
        df_test_all = pd.concat([df_test_all, pred,prob], axis=1)


    final_pred_list = []
    final_prob_list = []
    
    df_test_all.to_csv('result/{}/prediction_all.csv'.format(round_path), index=False)
    

    
# add class 0
label_dict = {'-2':0, '-1':1,'1':2, '2':3, '0':4, '5':5, '-3':7}
# label_list = ['-2', '-1', '1', '2', '0', '-3']
label_list = ['-2', '-1', '1', '2', '0', '5', '-3']
path = None
NP = ''

parser = ArgumentParser()
parser.add_argument("-r", "-- round times", dest="round_times")
parser.add_argument("-p", "-- assign a path under result ", dest="path")
parser.add_argument("-m", "-- is np model?", dest="np_model")

args = parser.parse_args()

if (args.round_times):
   round_times = int(args.round_times)
if (args.path):
   path = args.path
if (args.np_model):   
   np_model = args.np_model

if np_model == 'np':
   # NP = '_np'
   label_list = ['0', '5']
      
for n in range(round_times):  
    round_path = 'round_{}'.format(n) 
    print(n)
    testing(n, round_times, label_list, label_dict, path)


# def train_classifier(n, round_times, fileIndex, fileNum):
now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M')
print('Task start at: {}'.format(start_timestamp))
print('Task done at: {}'.format(timestamp))

