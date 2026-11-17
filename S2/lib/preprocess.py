# # Data preprocessing
# functions
import pandas as pd
import re
import csv
import os
import random
from argparse import ArgumentParser

label_dic = {'-2':0,'-1':1,'1':2,'2':3,'0':4, '5':5, '-3':7}

drop_segLen = 4
min_segLen = 17

def isNaN(num):
    return num != num
    
#===========================================================================================================
# split content and remove English
def split_content(content):
   
    split_re = r",|，|。|\s|，|,|？| |。|\n|\?|\.|\～|\?|！|!|;"
    replace_re = r",|，|。|\s|，|,|？| |。|\n|\(|\)|\?|\.|\～|\?|\<|\_|\>|\＞|…|、|\^|\~|：|:|\「|\」|\（|\）|\』|\『|！|\"|\t|；|\=|\“|\”|\;|\-|\→|\/|\※|\+|\《|\》|\\|[0-9]"
    filter_re = r"^http|^\d|^[a-zA-Z]*$|^\d"
    split = []
    
    if not isNaN(content):
      split = re.split(split_re, content)
      split = list(filter(lambda x: re.search(filter_re, x) is None, split))
    
    # remove the symbols in replace_re
    new_split = []
    for seg in split:  
        nseg = re.sub(replace_re,'',seg)
        #nseg = seg.replace(r'[^\u4e00-\u9FFF]','')
        nseg = nseg.strip(' ')
        if nseg != '':
          new_split += [nseg]
    
    return new_split
    
#=========================================================================================    
def separate_data(df_input, round_path, df_modify_p, sample_0_number, g_type, mix_non_answer = False, test_exclusive_0 = False, continous_answer = False, df_add_positive = None, df_add_negative = None):    
    # Process for all train data
    
    # data frame for storing training/tesing data
    df_out = pd.DataFrame({'id':[],'sentence':[],'label':[],'article_label':[]})
    
    # data frame for storing negative training data
    if g_type == 'train':
       df_neg = pd.DataFrame({'id':[],'sentence':[],'label':[],'article_label':[]})
    
    for i in range(df_input.shape[0]):
        content = df_input.at[i,'content']
        index = df_input.at[i,'id']
        answer = df_input.at[i,'態度分數句']
        if not(pd.isna(df_input.at[i,'S2'])):
           score = int(df_input.at[i,'S2'])
        else:   
           score = 0
        
        # split content and answer
        content = content.strip(' ')
        split = split_content(content)
           
        new_ans = ''
        new_ans_split = split_content(answer)
        
        ans_split = [i for i in new_ans_split if i !='']
        for ans in ans_split:
            new_ans += ans
            new_ans += ' '     
        
        # let every training data larger than min_segLen(20)
        # Each split will be the beginning, unless the remaining words less than 20
        sents = []
        
        for j in range(len(split)-1):
            sent = split[j]
            k = 0
            while len(sent) < min_segLen and j+1+k<(len(split)):
                    sent += (' ' + split[j+1+k])
                    k=k+1
            if len(sent) >= min_segLen:
                sents.append(sent)        
                
        # special process for the last sentence; if it is concated with the precious sentence, then append
        if len(split) >= 1:
           sent = split[-1]
        else:   
           sent = ''       
        if len(sents) >=1:
           pre_sent = sents[-1]
        else:   
           pre_sent =''
         
        if (pre_sent.find(sent) == -1) and (len(sent) > drop_segLen):
           sents.append(sent)        
       
        # set each substring's score and insert into the data frame 
      
        for s in sents:
            label = '0'  #defailt label
            if continous_answer:  # all answer sentence should be in s, label s with the score
                all_satisfy = True
                for j in new_ans_split:    
                    if (j != '') and not (j in s):  
                        all_satisfy = False
                        break
                if all_satisfy:
                   label = score
            else: # any answer sentence in s, label s with the score
                for j in new_ans_split:    
                    if (j != '') and (j in s):  
                       label=score 
            # check the segments in modified labels
            if str(label) == '0': # if not labeled by the precious given answer      
               result = df_modify_p
               for k in range(result.shape[0]):
                  modify_seg = result.at[k,'segment']
                  modify_label = result.at[k,'SegmentLable']
                  if continous_answer:  # all answer sentence should be in s, label s with the score
                     all_satisfy = True
                     for j in modify_seg.split(' '):
                         if (j != '') and not (j in s):  
                             all_satisfy = False
                             break
                     if all_satisfy:  
                         label = modify_label 
                  else: # any if any modify sentence in s, label s with the modify_label
                       for j in modify_seg.split(' '):
                          if (j != '') and (j in s): 
                              label = modify_label
                          
            if g_type == 'train':        
               if str(label) == '0':
                   df_neg.loc[df_neg.shape[0]] = [index,s,label,score]
               elif mix_non_answer:
                   df_out.loc[df_out.shape[0]] = [index,s,label,score]     
            else:
               if test_exclusive_0:
                  if str(label) != '0':  #only non_zero labeled data is used as test
                     df_out.loc[df_out.shape[0]] = [index,s,label,score]
               else:   
                  df_out.loc[df_out.shape[0]] = [index,s,label,score]
               
        # positive單獨輸出（ouly output the answer segment as a training example)
        if g_type == 'train' and mix_non_answer == False:  
           df_out.loc[df_out.shape[0]] = [index,new_ans,score,score]
    
    if not df_add_positive.empty:
       for k in range(df_add_positive.shape[0]):
            add_seg = df_add_positive.at[k,'segment']
            add_label = df_add_positive.at[k,'SegmentLable']
            df_out.loc[df_out.shape[0]] = ['add_positive',add_seg,add_label,add_label]  
                      
    if g_type == 'train':
       df_neg = df_neg.sample(n = sample_0_number, random_state = 0, replace= True)
       if not df_add_negative.empty:
          for k in range(df_add_negative.shape[0]):
              add_seg = df_add_negative.at[k,'segment']
              add_label = df_add_negative.at[k,'SegmentLable']
              df_neg.loc[df_neg.shape[0]] = ['add_negative',add_seg,add_label,add_label]  
       
       df_out = pd.concat([df_out, df_neg])         
       df_out.to_csv('data/data_4train/{}/train.csv'.format(round_path),encoding='utf-8_sig',index=False)
    elif g_type == 'valid':
       df_out.to_csv('data/data_4train/{}/test.csv'.format(round_path),encoding='utf-8_sig',index=False)
    elif g_type == 'test': 
       df_out.to_csv('data/data_4test/{}/test.csv'.format(round_path),encoding='utf-8_sig',index=False)
    
    return
    
#==========================================================================================    
def generate_classifier_data(label_list, round_path, g_type):    
    for label in label_list:
        #print(i, label)
        if g_type == 'train':
           f_name = 'data/data_4train/'+str(round_path)+'/train'+str(label_dic[label])+'.csv'
        elif g_type == 'valid':
           f_name = 'data/data_4train/'+str(round_path)+'/test'+str(label_dic[label])+'.csv'    
        elif g_type == 'test':
           f_name = 'data/data_4test/'+str(round_path)+'/test'+str(label_dic[label])+'.csv' 
        csvfile=open(f_name,'w',newline='',encoding='utf-8_sig')         
           
        writer=csv.writer(csvfile)
        writer.writerow(['id','sentence','score_label','label'])
        
        if g_type == 'train':
           f_source = 'data/data_4train/'+str(round_path)+'/train'+'.csv'
        elif g_type == 'valid':
           f_source = 'data/data_4train/'+str(round_path)+'/test'+'.csv'   
        elif g_type == 'test':
           f_source = 'data/data_4test/'+str(round_path)+'/test'+'.csv' 
    
        with open(f_source,encoding='utf-8_sig') as data_source:
                myCsv = csv.reader(data_source)
                header = next(myCsv)
                for row in myCsv:
                    #id = int(float(row[0]))
                    id = row[0]
                    if(row[3] == label):
                        new_id = 'p_' + str(id)
                    else:
                        new_id = 'n_' + str(id)
                        
                    if(int(float(row[2])) == int(label)):
                        new_label = 'positive'
                    else:
                        new_label = 'negative'
                    writer.writerow([new_id, row[1], row[2], new_label])            
        csvfile.close()                       
    return
        

#==========================================================================================    
def generate_p_n_pair_files(label_list, round_path, negative_sample_times):
    # For generate negative sample for computing contrastive loss
    for l in label_list :
        file_num = label_dic[l]
        df = pd.read_csv('data/data_4train/{}/train{}.csv'.format(round_path,file_num))
        df = df.drop(df.columns[0], axis=1)

        # generate positive samples
        df_positive = df[df['label'] == 'positive'] # original copy of positive samples                
        df_positive_all = df_positive
        
        n_sample_times = negative_sample_times[l] # get times of negative samples for each class label
        for i in range(n_sample_times-1):  # duplicate potitive n_sample_times
            df_positive_all = pd.concat([df_positive_all,df_positive])
        df_positive_all.to_csv('data/data_4train/{}/train{}_p.csv'.format(round_path,file_num),index=False)
       
        # generate negative samples       
        df_n = df[df['label'] == 'negative'] # original copy pf negative samples
      
        if l == '0':  # get negative samples for label '0'
           df_negative = df_n  # all negative samples
        else: # get negative samples for labels '-2, -1, +1, +2'
           df_negative = df_n[df_n['score_label'] != 0] # samples with non'0'
          
        default_neg_num = df_negative.shape[0]  
         
        df_negative_all = pd.DataFrame()
        df_negative_all = pd.concat([df_negative_all,df_negative]) # append default negative samples
        
        num_p_samples = df_positive_all.shape[0]
        add_n_samples = num_p_samples - default_neg_num  # number of needing additional negative sample 
        
        i = random.randint(1, 100)
        if add_n_samples >= 0: # number of default negative samples is less than positive samples, add from df_n
           add_negative_sample = df_n.sample(n=add_n_samples,random_state=i,replace=True)  # add replace=True
           df_negative_all = pd.concat([df_negative_all,add_negative_sample])
        else: # number of default negative samples is larger than positive samples, sample from df_negative
           add_negative_all = df_negative.sample(n=num_p_samples,random_state=i,replace=True)
             
        df_negative_all.to_csv('data/data_4train/{}/train{}_n.csv'.format(round_path,file_num),index=False)
    return
    
#=======================================================================================
def preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number, exclusive, continous_answer, np_model, test_model):
    for n in range(round_times):
      if np_model == 'np':
         score = 'NP_Label'
      else:
         score = 'S2'   
        
      # read data
      round_path = 'round_{}'.format(n)
       
      if gen_type == 'train':
         os.system('mkdir -p data/data_4train/{}'.format(round_path))    
         # read training data
         df_train = pd.read_csv('{}/train{}.csv'.format(aspect,n+1))
         df_train_p = df_train
         # get required information, and reset index and name
         df_train_p = df_train_p.reset_index() 
         df_train_p = df_train_p.loc[:,['id','content','態度分數句',score]]
         df_train_p.rename(columns = {score:'S2'}, inplace = True)
          
         
      elif gen_type == 'test': 
         os.system('mkdir -p data/data_4test/{}'.format(round_path))  
    
      # read testing data   
      if test_model == 'combine_article' :
        df_test = pd.read_csv('{}/prediction{}.csv'.format(aspect,n))
      else:
        df_test = pd.read_csv('{}/test{}.csv'.format(aspect,n+1))
      df_test_p = df_test
      # get required information, and reset index and name
      df_test_p = df_test_p.reset_index()
      if test_model == 'combine_article' :
        df_test_p = df_test_p.loc[df_test_p['model_pred'] != 0,['id','content','態度分數句',score]]
        df_test_p.rename(columns = {score:'S2'}, inplace = True)
        df_test_p = df_test_p.reset_index(drop = True)  
      else:
        df_test_p = df_test_p.loc[:,['id','content','態度分數句',score]]
        df_test_p.rename(columns = {score:'S2'}, inplace = True)
      
    
      # read additional modified labels:
      df_modify_p = pd.read_csv('{}/modify.csv'.format(aspect)) 
      # read individual positive label segments:
      df_add_positive = pd.read_csv('{}/add_positive.csv'.format(aspect)) 
      # read individual negative label segments:
      df_add_negative = pd.read_csv('{}/add_negative.csv'.format(aspect)) 
      
      if gen_type == 'train':
         separate_data(df_train_p, round_path, df_modify_p, sample_0_number, g_type = 'train', mix_non_answer = True, continous_answer = continous_answer,                           df_add_positive = df_add_positive, df_add_negative = df_add_negative)
         separate_data(df_test_p, round_path, df_modify_p, sample_0_number, g_type = 'valid', test_exclusive_0 = exclusive, continous_answer = continous_answer, df_add_positive = df_add_positive, df_add_negative = df_add_negative)
      elif gen_type == 'test':   
         separate_data(df_test_p, round_path, df_modify_p, sample_0_number, g_type = 'test', test_exclusive_0 = False, continous_answer = continous_answer, df_add_positive = df_add_positive, df_add_negative = df_add_negative)
        
      # Train
      # get each substring is P or N in different scores
      if gen_type == 'train':
         generate_classifier_data(label_list, round_path, g_type = 'train')
         generate_classifier_data(label_list, round_path, g_type = 'valid')
      # Test
      elif gen_type == 'test':
       generate_classifier_data(label_list, round_path, g_type = 'test')     
       
    print('Preprocess stage1 done.')
    return
    
def preprocess_stage2(label_list, gen_type, round_times, negative_sample_times):
    if gen_type == 'train':    
       for n in range(round_times):    
         round_path = 'round_{}'.format(n)        
         generate_p_n_pair_files(label_list, round_path, negative_sample_times)        
       print('Preprocess stage2 done.')
    return 
       
