import pandas as pd
from os.path import exists
from os import mkdir
import csv

def sample_data(df_positive, df_negative, base_size, PN_ratio, random_state, pred_wrong):
    
    # for ensemble model
    
    #print('pred wrong index:{}'.format(pred_wrong))

    positive_id = df_positive.loc[:, 'sent_id'].to_list()
    negative_id = df_negative.loc[:, 'sent_id'].to_list()
    
    positive_weights = [1]*df_positive.shape[0]
    negative_weights = [1]*df_negative.shape[0]
    
    for i in range(len(positive_id)):
        if positive_id[i] in pred_wrong:
            #print(i,positive_id[i])
            positive_weights[i] = 100
        
    for i in range(len(negative_id)):
        if negative_id[i] in pred_wrong:
            #print(i,negative_id[i])
            negative_weights[i] = 100

    positive_sample_size = base_size*PN_ratio[0]
    negative_sample_size = base_size*PN_ratio[1]

    df_positive_sample = df_positive.sample(n = positive_sample_size, random_state=random_state, weights = positive_weights)
    
    df_negative_sample = df_negative.sample(n = negative_sample_size, random_state=random_state, weights = negative_weights)

    return pd.concat([df_positive_sample, df_negative_sample],axis = 0)


        
def voting(pred_list,train_acc_list = []):
    
    # for ensemble model
    # pred_list: a list of N results of predictions, N = the number of models.
    data_num = len(pred_list[0])
    final_pred_list = []

    for i in range(data_num):
        tmp_list = []
        for e  in pred_list:
            tmp_list.append(e[i])
        final_pred = (max(tmp_list,key=tmp_list.count))
        final_pred_list.append(final_pred)
        
    return final_pred_list



def save_scores_table(scores, OUTPUT_FILE, multi_model=False):
    if multi_model:
        with open(OUTPUT_FILE, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['model','accuracy=micro f1','macor f1','P-precision','P-recall','P-f1','N-precision','N-recall','N-f1'])
            
            model_label = ['-2','-1','1','2','0']
            
            for i,score in enumerate(scores):
                
                acc = round(score[0],3)
                
                p_pre = round(score[1][1],3)
                n_pre = round(score[1][0],3)

                p_rec = round(score[2][1],3)
                n_rec = round(score[2][0],3)

                p_f1 = round(score[3][1],3)
                n_f1 = round(score[3][0],3)

                mac_f1 = round((p_f1+n_f1)/2,3)
                writer.writerow([model_label[i],acc,mac_f1,p_pre,p_rec,p_f1,n_pre,n_rec,n_f1])
        
        
    else:
        with open(OUTPUT_FILE, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['accuracy=micro f1','macor f1','P-precision','P-recall','P-f1','N-precision','N-recall','N-f1'])

            acc = round(scores[0],3)

            p_pre = round(scores[1][1],3)
            n_pre = round(scores[1][0],3)

            p_rec = round(scores[2][1],3)
            n_rec = round(scores[2][0],3)

            p_f1 = round(scores[3][1],3)
            n_f1 = round(scores[3][0],3)

            mac_f1 = round((p_f1+n_f1)/2,3)

            writer.writerow([acc,mac_f1,p_pre,p_rec,p_f1,n_pre,n_rec,n_f1])
