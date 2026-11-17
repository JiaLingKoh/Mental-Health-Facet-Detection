import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer
#from IPython.display import clear_output
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from transformers import BertConfig,BertModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt       
from sklearn.metrics import confusion_matrix

class SentenceDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    #def __init__(self, mode, tokenizer,fileIndex):
    def __init__(self, mode, tokenizer,df):
        assert mode in ["train", "test","testtest"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        #self.df = pd.read_csv('./tsv/'+mode+str(fileIndex) + ".tsv", sep="\t").fillna("")
        self.df = df
        
        self.len = len(self.df)
        self.label_map = {'positive': 1, 'negative': 0,'test': 9}
        #self.label_map = {'-2': 0, '-1': 1, '1':2, '2':3}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "testtest":
            sentence,_ = self.df.iloc[idx,:].values
            label_tensor = None
        else:
            sentence, label = self.df.iloc[idx, :].values
            label = str(label)
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
           
        # 建立第一個句子的 BERT tokens //並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        sentence = self.tokenizer.tokenize(sentence)
        word_pieces += sentence + ["[SEP]"]
        len_a = len(word_pieces)
         
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a , 
                                        dtype=torch.long)
        return (tokens_tensor,segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len



# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_predictions(model, dataloader, act_fun = None, use_threshold = False, threshold = None):
    predictions = None
    correct = 0
    p_correct = 0
    total = 0
    
    probs = None
    positive_num = 0
    predict_positive = 0  
    
    with torch.no_grad():
        # 遍巡整個資料集
        for i, data in enumerate(dataloader):
#             if (i % 50) == 0:
#                 print("Progress ... batch",i) 
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            
            tokens_tensors, segments_tensors, masks_tensors, labels = data[:4]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,
                            )
            
            logits = outputs[0]
            p_values = logits
            #print(logits.data)
            
            if act_fun == 'sigmoid':
               p_values = torch.sigmoid(logits)
            elif act_fun == 'softmax':   
               m = nn.Softmax(dim=1)
               p_values = m(logits)  
            
            if use_threshold:
               threshold_value = threshold
               threshold = torch.tensor([threshold_value])
               threshold=threshold.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
               pred = ((p_values.data > threshold).int())[:,-1]
               prob = p_values.data
              
               #_, pred = torch.max(logits.data, 1)
            
            
            else: 
               prob, pred = torch.max(p_values.data, 1)
            
            # 用來計算訓練集的分類準確率
            
            labels = data[3]
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            for i, l in enumerate(labels):
               if (l==1):
                  positive_num += 1
                  if labels[i] == pred[i]:
                     p_correct += 1
            for l in pred:
               if (l==1):
                  predict_positive += 1      
            #print("Pred:",pred)
            #print("Prob:",prob)
            #print(labels)
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
                
            if probs is None:
                probs = prob
            else:
                probs = torch.cat((probs, prob))
            
                 
    acc = correct / total
    if predict_positive!= 0:
       precision = p_correct / predict_positive
    else:
       precision = 0
    if positive_num!= 0:   
       recall = p_correct / positive_num
    else:
       recall = 0   
    
    return predictions, probs, acc, precision, recall
 
def trans(x):
    
    if x == 'positive':
        return 1
    else:
        return 0
