import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertConfig,BertModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
#import matplotlib.pyplot as plt
    
from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import csv
import datetime

# import custom library
from lib.utils import *
from lib.model import *


torch.manual_seed(0)

PRETRAINED_MODEL_NAME = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

now = datetime.datetime.now()
start_timestamp = now.strftime('%Y-%m-%d %H:%M')

print("======")
print("PyTorch Version：", torch.__version__)
print("CUDA Version：", torch.version.cuda)
print("======")

torch.cuda.empty_cache()

pred_list = []
train_acc_list = []
best_epoch_list = []

# number of models
fileNum = 5

train_type = '5_model'
equal_epoch = True
round_times = 5
EPOCH_NUM = 15
loss_fun = 'bce&supervised_contrastive_loss'
data_dic = "data/data_4train"

# batch 是否 shuffle
batch_shuffle= True
# 每個model預測的分類數
NUM_LABELS = 2

s_indexes = ['precision','f1']

# BCE的比例
W_bce_p = [1, 3, 3, 1, 2, 5, 1]
W_bce_n = [1, 1, 1, 1, 1, 1, 1]
loss_margin = 0.5

# BCE : margin 的比例 a : 1-a
W_bce_margin = [0.3, 0.5, 0.5, 0.5, 0.75, 0.5, 0.5]
train_batch_sizes = {"-2": 4, "-1": 4, "+1": 4, "+2": 4, "0": 16, "+5":4, "-5":4}

torch.cuda.empty_cache()

if not os.path.exists('picture'):
    os.mkdir('picture')

def sub_image(fig, x, y, Type='BCE'):
    fig.plot(x, y)
        
    fig.set_title(Type) # title
    
    return fig

def f1_image(epoch, f1, fileIndex):
    fig, m1 = plt.subplots()

    m1 = sub_image(m1, epoch, f1, "F1")

    if not os.path.exists('picture'):
        os.mkdir('picture')
    
    plt.savefig('picture/{}_f1.png'.format(fileIndex))

def loss_image(batch, BCE, BCE2, margin, epoch):
    fig, (m1, m2, m3) = plt.subplots(3, 1, sharex='row')

    m1 = sub_image(m1, batch, BCE, "BCE")
    m2 = sub_image(m2, batch, BCE2, "BCE2")
    m3 = sub_image(m3, batch, margin, "margin")
    
    plt.savefig('picture/Epoch {}.png'.format(epoch))
    
def Loss_image(batch, loss, epoch):
    fig, m1 = plt.subplots()

    m1 = sub_image(m1, batch, loss, "Loss")
    
    plt.savefig('picture/Epoch {}_loss.png'.format(epoch))

    
def Losses_image(batch, BCE, BCE2, margin, loss, epoch):
    fig, (m4, m1, m2, m3) = plt.subplots(4, 1, sharex='row')

    m4 = sub_image(m4, batch, loss, "Loss")
    m1 = sub_image(m1, batch, BCE, "BCE")
    m2 = sub_image(m2, batch, BCE2, "BCE2")
    m3 = sub_image(m3, batch, margin, "margin")
    
    plt.savefig('picture/Epoch {}_Losses.png'.format(epoch))

def train_classifier(n, round_times, fileIndex, fileNum, batch_size):
    f1_list = []
    epoch_list = []
    BATCH_SIZE = batch_size
    print('round: {}/{}, model: {}/{}'.format(n+1,round_times,fileIndex+1,fileNum)) 

    # load train
    df_train = pd.read_csv('{}/{}/train{}_p.csv'.format(data_dic,round_path,fileIndex))
    df_train = df_train.loc[:, ['sentence','label']]

    df_train2 = pd.read_csv('{}/{}/train{}_n.csv'.format(data_dic,round_path,fileIndex))
    df_train2 = df_train2.loc[:, ['sentence','label']]

    # load test
    df_test  = pd.read_csv('{}/{}/test{}.csv'.format(data_dic,round_path,fileIndex))
    df_test = df_test.loc[:, ['sentence','label']]

    trainset = SentenceDataset("train", tokenizer=tokenizer, df=df_train)
    trainset2 = SentenceDataset("train", tokenizer=tokenizer, df=df_train2)
    testset = SentenceDataset("test", tokenizer=tokenizer, df=df_test)

    # 初始化一個每次回傳 BATCH_SIZE 個訓練樣本的 DataLoader
    # 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=batch_shuffle, collate_fn=create_mini_batch)
    trainloader2 = DataLoader(trainset2, batch_size=BATCH_SIZE, shuffle=batch_shuffle, collate_fn=create_mini_batch)
    # 載入一個可以做中文多分類任務的模型，n_class = 2

    PRETRAINED_MODEL_NAME = "bert-base-chinese"

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = model.to(device)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # change to 1e-6?
    
    best_epoch = 0
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    best_avg = 0

    if equal_epoch:
        EPOCHS = EPOCH_NUM

    train_f1_list = []
    test_f1_list = []

    for epoch in range(EPOCHS):
        N_batch = 0
        batch_list = []
        BCE_list = []
        BCE2_list = []
        margin_list = []
        loss_list = []
        has = False
        print("Epoch:---", epoch+1)
         # training mode
        model.train()

        running_loss = 0.0
        for data,data2 in zip(trainloader,trainloader2):
            N_batch = N_batch + 1

            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            tokens_tensors2, segments_tensors2, masks_tensors2, labels2 = [t.to(device) for t in data2]            
                
            # 將參數梯度歸零
            optimizer.zero_grad()

            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,
                            labels=labels
                            )
            logits = torch.sigmoid(outputs[1]).to(torch.float32) 
            #logits = torch.sigmoid(outputs[1])      
            outputs2 = model(input_ids=tokens_tensors2, 
                             token_type_ids=segments_tensors2, 
                             attention_mask=masks_tensors2,
                             labels=labels2
                             )                
            logits2 = torch.sigmoid(outputs2[1]).to(torch.float32)
            #logits2 = torch.sigmoid(outputs2[1])
                         
            BCE_loss = outputs[0].item()*W_bce_p[fileIndex] + outputs2[0].item()*W_bce_n[fileIndex]

                
            # Margin Loss              
                   
            Margin_loss = []
            l = torch.tensor([1]*(logits.shape[1]))        
            loss_fn = nn.MarginRankingLoss(margin=loss_margin)
            for input1,input2,target in zip(logits[:,1],logits2[:,1],l):
                
                if loss_fun == 'margin_loss' or loss_fun == 'bce&margin_loss':
                    loss_result = loss_fn(input1,input2,target)
                    Margin_loss.append(loss_result)
                elif loss_fun == 'bce&supervised_contrastive_loss':
                    exp_input1 = torch.exp(input1)
                    exp_input2 = torch.exp(input2)
                    loss_result = -1 * torch.log(exp_input1 / (exp_input1 + exp_input2))
                    Margin_loss.append(loss_result)
#                     print('1:{}  2:{}'.format(input1, input2))
#                     print('1:{}  2:{}'.format(exp_input1, exp_input2))
#                     print(loss_result)
#                     print(input1)


            
            if loss_fun == 'bce_loss':
                loss = BCE_loss
            elif loss_fun == 'margin_loss':       
                loss = sum(Margin_loss) / len(Margin_loss)
            elif loss_fun == 'bce&margin_loss':   
                loss = BCE_loss * W_bce_margin[fileIndex] + (sum(Margin_loss) / len(Margin_loss)) * (1-W_bce_margin[fileIndex])
#                 print('P:{}  N:{}  BCE:{}'.format(outputs[0].item(),outputs2[0].item(),BCE_loss)) 
#                 print('Margin_loss:{}'.format(sum(Margin_loss) / len(Margin_loss)))
            elif loss_fun == 'bce&supervised_contrastive_loss':
                loss = BCE_loss * W_bce_margin[fileIndex] + (sum(Margin_loss) / len(Margin_loss)) * (1-W_bce_margin[fileIndex])
#                 print('P:{}  N:{}  BCE:{}'.format(outputs[0].item(),outputs2[0].item(),BCE_loss)) 
#                 print('Margin_loss:{}'.format(sum(Margin_loss) / len(Margin_loss)))

            
            loss_list.append(loss.item())
            batch_list.append(N_batch)
            BCE_list.append(outputs[0].cpu().item())
            BCE2_list.append(outputs2[0].cpu().item())
            margin_list.append((sum(Margin_loss) / len(Margin_loss)).item())
                
            # backwards
            loss.backward()
            optimizer.step()

            # 紀錄當前 batch loss
            running_loss += loss.item()

        # 計算分類準確率
        _, _, acc, precision, recall = get_predictions(model, trainloader)

        # Test ================================

        # testing mode
        model.eval()

        testloader = DataLoader(testset, batch_size= BATCH_SIZE , collate_fn=create_mini_batch) 
        _,  _, test_acc, test_precision, test_recall = get_predictions(model, testloader, act_fun = 'sigmoid', use_threshold = False, threshold = 0.5)

        print('[epoch %d] loss: %.3f, train_precision: %.3f, train_recall: %.3f, test_precision: %.3f, test_recall: %.3f'
         % (epoch+1, running_loss, precision, recall, test_precision, test_recall))
        
        if  (precision + recall) != 0:   
            f1 = 2 * (precision * recall)/(precision + recall)   
        else:  
            f1 = 0    
        
        if  (test_precision + test_recall) != 0:      
            test_f1 =  2 * (test_precision * test_recall)/(test_precision + test_recall)  
        else:     
            test_f1 = 0
        test_avg = (test_precision + test_recall)/2  
            
        for select_index in s_indexes:
            best = False
            if ((epoch >= 1) or (test_recall > 0.5)):  # to prevent the outlier with recall is very low at the first epoch
               if select_index == 'f1' :
                  if test_f1 > best_f1: 
                     best_f1 = test_f1
                     best = True
               elif select_index == 'precision':  
                  if (test_precision > best_precision) or ((test_precision == best_precision) and (test_recall > best_recall)):
                     best_precision = test_precision
                     best_recall = test_recall
                     best = True         
           
            if best:  # save the best model 
               best_epoch = epoch + 1
               
               os.system('mkdir -p model_{}/{}'.format(select_index,round_path))
               torch.save(model.state_dict(),'model_{}/{}/{}.ckpt'.format(select_index, round_path,str(fileIndex))) 
               # only save best to prevent output memory exceed error     
               print('model_{}: {}.ckpt saved.'.format(select_index,fileIndex))
                
        f1_list.append(test_f1)
        epoch_list.append(epoch+1)

        train_f1_list.append(f1)
        test_f1_list.append(test_f1)
        
    f1_image(epoch_list, f1_list, fileIndex)
    if select_index == 'f1':
       best_epoch_list.append({'model':fileIndex,'best_f1':best_f1,'best_epoch':best_epoch})
    elif select_index == 'precision':
       best_epoch_list.append({'model':fileIndex,'best_precision':best_precision,'best_epoch':best_epoch})  
    elif select_index == 'p_r_avg':
       best_epoch_list.append({'model':fileIndex,'best_p_r_avg':best_avg,'best_epoch':best_epoch})   
    
    print('model {} [P: {}, N: {}, B_M: {}]'.format(fileIndex, W_bce_p[fileIndex], W_bce_n[fileIndex], W_bce_margin[fileIndex]))
    return
        
#=================================================================================================
score_name = {"-2":0,"-1":1,"+1":2,"+2":3,"0":4, "+5":5, "-5":6}
train_models = ["-2","-1","+1","+2","0"]
#train_models = ["-2","-1","+1","+2"]
# train_models = ["0", "+5"]
#train_models = ["-3"]

parser = ArgumentParser()
parser.add_argument("-r", "-- round times", dest="round_times")
parser.add_argument("-m", "-- trained model", dest="train_models")
parser.add_argument("-a", "-- aspect", dest="aspect")
parser.add_argument("-p", "-- train type", dest="train_type")

args = parser.parse_args()
if (args.round_times):
   round_times = int(args.round_times)
if (args.train_models):   
   train_models_str = args.train_models
   train_models = train_models_str.split(' ')
if (args.aspect):   
   aspect = args.aspect
if (args.train_type):   
   train_type = args.train_type

if train_type == 'np':
#     train_models = ["0", "+5"]
    train_models = ["+5"]

print(train_type)

for m in train_models:
    for n in range(round_times):  
        round_path = 'round_{}'.format(n) 
        torch.manual_seed(0)
        fileIndex = score_name[m]
        print(fileIndex)
        fileNum = len(train_models)
        batch_size = train_batch_sizes[m]
        
        train_classifier(n, round_times, fileIndex, fileNum, batch_size)

    print('best_epoch:') 
    for e in best_epoch_list:
        print(e)

now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M')
print("Task start at: {}".format(start_timestamp))
print('Task done at: {}'.format(timestamp))

