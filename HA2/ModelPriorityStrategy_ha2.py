
# for model priority
import pandas as pd
import numpy as np

# for metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import csv
from argparse import ArgumentParser
from ResultsParsing import out_results

model_priority = ['0','2','-2','-1','1','-3']

model_to_num = {'-2':0, '-1':1, '+1':2, '+2':3, '0':4, '-3':7}

labels = [-2, -1, 1, 2, 0]

# 計算分數、產生混淆矩陣、輸出Segment、Article的分數csv file

def draw_confusion_matrix(matrix, outpath):
    # create figure and axes
    fig, ax = plt.subplots()

    # show confusion matrix
    im = ax.matshow(matrix, cmap=plt.cm.winter)

    # set labels for axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # add annotations to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", color="w")

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set title and labels
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.savefig(outpath)
    return
    
def metric_scores(y_true,y_pred,round_num,mode,Results_FolderName):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # compute confusion matrix with specified label order
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    cm_outfile = 'result/{}round_{}_cm.png'.format(Results_FolderName,round_num)
    draw_confusion_matrix(cm, cm_outfile)
    #plt.show()

    # Output score
    with open('result/{}round_{}_{}Score.csv'.format(Results_FolderName,round_num,mode),'w') as file:
        writer = csv.writer(file)
        writer.writerow(['class','accuracy','precision','recall','f1-score'])
        for i,l in enumerate(labels):
            writer.writerow([l, round(accuracy,3), round(precision[i],3), round(recall[i],3), round(f1[i],3)])

    return cm

# Majority test
def similar_frquency(test_value, compare_value):
    if test_value > compare_value:
        if test_value/compare_value < 1.2:          
          return True
        else:
          return False
    else:
        if compare_value/test_value < 1.2:          
          return True
        else:
          return False      
          
# Majority test
def majorty(test_value, compare_value):
    if compare_value == 1:
       if test_value >= 2:
          return True
       else:
          return False
    else:
       if test_value/compare_value > 1.5:          
          return True
       else:
          return False
          
# decide article label
def det_article_label(p_id, seg_labels, article_decision):   
    lab_counts = {-2:[0,-1], -1:[0,-1], 1:[0,-1], 2:[0,-1],-3:[0,-1]}
    non_zero = False    
    Label = 0
            
    for i, l in enumerate(seg_labels):             
         #for l in ListOfSegmentLabelForModel: # 找出該序列中非0的最後一個，e.g. [2,1,1,0,1,-1,0,0,0]，輸出為-1
         if l !=0 :
             non_zero = True
             if article_decision == 'last':  
                Label = l # 決定PO文的label : 該篇PO文最後一個segment，被哪一個分類器先選走，就屬於該分類器
             elif article_decision in ['voting']:
                   lab_counts[l][0] += 1 
             elif article_decision in ['v&last', 'priority','rule']:      
                   LabelForLast = l
                   lab_counts[l][0] += 1 
                   lab_counts[l][1]= i   
                                   
    if not non_zero:
        Label = 0 # 決定PO文的label (all are 0s)        
    elif article_decision == 'voting':
        Label = max(lab_counts, key= lambda key: lab_counts[key][0])  # the label with the maximum count 
    elif article_decision == 'v&last':
        Label = max(lab_counts, key= lambda key: lab_counts[key]) 
    elif article_decision == 'priority':
        p_count = lab_counts[2][0] + lab_counts[1][0]
        n_count = lab_counts[-2][0] + lab_counts[-1][0]
        if lab_counts[-2][0] > 0:
            Label = -2
        elif lab_counts[2][0] > 0:    
            Label = 2    
        elif lab_counts[1][0] > 0:
            Label = 1    
        elif lab_counts[-1][0] > 0:    
            Label = -1    
        else:
            Label = 0   
            
    elif article_decision == 'rule':
        p_count = lab_counts[2][0] + lab_counts[1][0]
        n_count = lab_counts[-2][0] + lab_counts[-1][0] + lab_counts[-3][0]
                 
        # Rules of deciding -2
        if (lab_counts[-2][0] > 0 and lab_counts[-1][0] > 0) and p_count == 0: # detect -2 and -1 and no positives
             if majorty(lab_counts[-1][0],lab_counts[-2][0]):  # -1 >> -2
                Label = -1  
             elif lab_counts[-1][1] > lab_counts[-2][1]: 
                Label = -1
             else:
                Label = -2          
        elif (lab_counts[-2][0] > 0 and p_count == 0):  # only detect -2 and no positives   
              Label = -2  
        elif (lab_counts[-2][0] > 0 and lab_counts[1][0] > 0) and majorty(lab_counts[-2][0],lab_counts[1][0]) : 
             # detect both -2 and 1 and -2 >> 1
                Label = -2  
                      
        # Rules of deciding +2        
        elif (lab_counts[2][0] > 0 and lab_counts[1][0] > 0) and n_count == 0: # detect both 2 and 1 and no negatives
             if majorty(lab_counts[2][0],lab_counts[1][0]):  # 2 >> 1
                Label = 2    
             elif majorty(lab_counts[1][0],lab_counts[2][0]):  # 1 >> 2
                Label = 1        
             elif lab_counts[1][1] > lab_counts[2][1]: # 1 appears after 2
                Label = 1
             else:
                Label = 2                                  
        elif (lab_counts[2][0] > 0) and lab_counts[-1][0] > 0: # detect both 2 and -1
             if majorty(lab_counts[2][0],lab_counts[-1][0]):  # 2 >> -1
                Label = 2
             elif majorty(lab_counts[-1][0],lab_counts[2][0]):  # -1 >> 2
                if p_count > n_count:
                   Label = 1 
                else:   
                   Label = -1      
             elif similar_frquency(lab_counts[2][0],lab_counts[-1][0]):
                if lab_counts[-1][1] > lab_counts[1][1]:
                   Label = -1    
                else:
                   Label = 1   
             elif lab_counts[-1][1] > lab_counts[2][1]: # -1 appears after 2
                Label = -1
             else:
                Label = 2        
        elif (lab_counts[2][0] > 0) and lab_counts[-3][0] > 0: # detect both 2 and -3
             if majorty(lab_counts[2][0],lab_counts[-3][0]):  # 2 >> -3
                Label = 2
             else:
                Label = 1
        elif lab_counts[2][0] > 0:
            Label = 2        
                                                  
        # Rules of deciding -1     
        elif (lab_counts[-1][0] > 0 and p_count == 0): # detect only -1
            Label = -1    
        # Rules of deciding +1
        elif (lab_counts[1][0] > 0 and n_count == 0): # detect only 1
            Label = 1               
        #  Rules of deciding -1/+1    
        elif (lab_counts[-1][0] > 0 and lab_counts[1][0] > 0): # detect both -1 and 1 
            if majorty(lab_counts[-1][0],lab_counts[1][0]): # -1 >> 1
               Label = -1
            elif majorty(lab_counts[1][0],lab_counts[-1][0]): # 1 >> -1
               if lab_counts[-1][1] > lab_counts[1][1]:
                  Label = -1
               else:
                  Label = 1      
            elif lab_counts[-1][1] > lab_counts[1][1]: # -1 appears after 1
               Label = -1 
            else:
               Label = 1     
        # otherwises      
        else:
            Label = max(lab_counts, key= lambda key: lab_counts[key][1])  # the label last appears
            
            if (Label == -3 and lab_counts[2][0] > 0):
               Label == 1
            elif (Label == -3 and (lab_counts[-1] > lab_counts[1])):
               Label = -1
            elif (Label == -3 and lab_counts[1] > lab_counts[-1]):
               Label = 1   
            else:   
               Label = 0          
    return Label          
        
# 用模型的優先權序列調整所有segment最後output的label
def ModelPriorityClassifier(Results_FolderName,round_num, except_0 = True, article_decision = None):
    """
    1. 先取得model 優先權序列
    2. 以 prediction_all.csv 得出有幾篇PO文
    3. 若一篇PO文切分為10句，因為有五個model，則每一個segment會有五個分數，e.g. [0 0 0 0 1]
    4. 以模型優先權序列尋找[0 0 0 0 1]中， 1 所對應的 model 編號。若segment的分數為 [0 0 0 0 0]，則判斷為 model 0
    5. 得到一篇PO文中，10個被模型優先權序列判斷後的 label，e.g. [0,1,0,-1,2,0,-1,2,0,0]
    6. 以序列中最後一個非0的label當作該篇PO文的label , 以(5)的範例來看，輸出為 2
    7. 得到所有PO文的label，跟原始資料集prediction_all.csv 的article_label算出混淆矩陣和分數
    8. segment label也要跟原始資料集prediction_all.csv 的 segment label 算出混淆矩陣和分數
    """
    

    df_prediction_all = pd.read_csv('result/{}round_{}/prediction_all.csv'.format(Results_FolderName,round_num))
    segment_id_list = list(df_prediction_all['id'])
    segment_list = list(df_prediction_all['sentence'])
    post_id_list = list(df_prediction_all['id'].dropna(axis=0, how='any').unique())   # 取出這一個資料集上有多少篇ＰＯ文
    
    ListOfPostLabel = [] # 收集原始資料集中所有PO文的label，e.g. 若有48篇，則有48個label
    ListOfSegmentLabel = [] # 收集原始資料集中，該篇PO文所有segment的label，e.g.48篇ＰＯ文切分成1000句，就有1000個label
    ListOfSegmentLabelForModel = [] # 收集所有segment被分類的label，
    ListOfLabelForArticle = [] # 該篇PO文中最後一個segment，所對應的分類器編號

    # post level
    
    for post_id in post_id_list:    # 這個資料集中有48篇不重複ID的PO文，會跑48次 , article level 
        # define for segment
        LastOfModelNameForSegment = '' 
  
        #post_label = df_prediction_all.loc[df_prediction_all['id']== post_id]['article_label'] # get id=50 的 文章label
        post_label = df_prediction_all[df_prediction_all['id'] == post_id]['article_label'].unique()[0]
        
        if except_0:
           if post_label == '0':
              print(post_id," is 0!")
              continue # skip the post with label 0
        
        ListOfPostLabel.append(post_label)  # 存文章的label
        
        df_post_to_segment = df_prediction_all[df_prediction_all['id'] == post_id] 
        # 取出所有相同 id 的segment , e.g. 文章id=100 , 且該篇PO文拆成10個segment, id都是100，則df_post有10筆segment 
        
        df_post_to_segment = df_post_to_segment.reset_index() 
        df_post_to_segment = df_post_to_segment.to_dict(orient='index') # all segments convert to dict , key是編號，從0開始
        seg_num = len(df_post_to_segment)
        
        # segment level
        non_zero = False
        for IndexOfSegment in df_post_to_segment: # 這篇po文有幾個segment就跑幾次
            #print(df_post_to_segment[IndexOfSegment]['sentence'])
            ModelForSegmentLabelFlag = False # check the segment's label

            ListOfSegmentLabel.append(df_post_to_segment[IndexOfSegment]['label']) # 收集原始資料集中，該篇PO文所有segment的label
            ListModelForSegmentLabel = []
        
            segment_dict = df_post_to_segment[IndexOfSegment] # IndexOfSegment : 如果一篇文章有10個segment，則IndexOfSegment會從0~9
            #print(segment_dict)

            
            for modelIndex in model_priority: # 五個分類器，跑五次
                #modelNum = model_to_num[modelIndex] # 分類器 0,2, - 2, 1,-1
                model_for_segment_label = segment_dict['model_for_{}'.format(modelIndex)] # 該分類器對此segment的label
                #print(modelIndex,":",model_for_segment_label)
                ListModelForSegmentLabel.append(model_for_segment_label) # list中 包含五個分類器對這句segment的label, e.g. [0 0 1 0 0]
            

            # if post_id == 6: # 假如要指定post id 可以用這行
            """以模型優先權判斷segment label"""
            
            for c_num,label in enumerate(ListModelForSegmentLabel): # list中 包含五個分類器對這句segment的label, e.g. [0 0 1 0 0]
                # print(df_post_to_segment[IndexOfSegment]) # 看各個segment的結果

                if label == 1: # 如果有在五個分類器中按照順序，找到1，則return一個index, 去對應 df_model_P_f1_sort 的分類器號碼
                    # print("index : "+str(_) + "   => " +str(int(df_model_P_f1_sequence['model'][_])))
                    LabelForSegment = int(model_priority[c_num]) # 單一segment 的 label 
                    
                    ListOfSegmentLabelForModel.append(LabelForSegment)  # 收集所有segment被分類的label
                    ModelForSegmentLabelFlag = True # flag set true
                    break # 碰到list中第一個 1 ，則停止尋找
                
                elif c_num == len(model_priority)-1 and ModelForSegmentLabelFlag == False: # 如果有segment的分數為[0 0 0 0 0]，則被判斷為 label 0
                    LabelForSegment = 0 
                    ListOfSegmentLabelForModel.append(LabelForSegment)  # 收集所有segment被model分類的label
        
                   
        seg_labels = ListOfSegmentLabelForModel[(seg_num*-1):]
        
        """decide article label"""
        LabelForArticle = det_article_label(post_id, seg_labels, article_decision) # determine article label according to seg_labels  
        
        ListOfLabelForArticle.append(LabelForArticle) # 最後句子被分類器判斷的結果，存為文章的label
    

    """for segment"""
    df_segment_id = pd.DataFrame(segment_id_list,columns=['id'])
    df_segment = pd.DataFrame(segment_list,columns=['segment'])
    df_SegmentLabel = pd.DataFrame(ListOfSegmentLabel,columns=['label'])
    df_SegmentLabelFromModel = pd.DataFrame(ListOfSegmentLabelForModel,columns=['ModelPred_SegmentLabel'])
    df_Segment_result = pd.concat([df_segment_id,df_segment, df_SegmentLabel,df_SegmentLabelFromModel],axis=1)
    seg_cm = metric_scores(y_true=ListOfSegmentLabel,y_pred=ListOfSegmentLabelForModel,round_num=round_num,mode="Segment",Results_FolderName=Results_FolderName)
    df_Segment_result.to_csv('result/{}round_{}_SegmentLabel.csv'.format(Results_FolderName,round_num))

    """for article"""
    df_id = pd.DataFrame(post_id_list,columns=['id'])
    df_PostLabelFromModel = pd.DataFrame.transpose(pd.DataFrame([ListOfLabelForArticle],index=['ModelPred_articleLabel']))       
    df_ListOfPostLabel = pd.DataFrame(ListOfPostLabel, columns=['article_label']) # 原始資料集的article label
    df_post_result = pd.concat([df_id,df_ListOfPostLabel,df_PostLabelFromModel],axis=1)
    article_cm = metric_scores(y_true=df_ListOfPostLabel,y_pred=df_PostLabelFromModel,round_num=round_num,mode="Article",Results_FolderName=Results_FolderName)
    df_post_result.to_csv('result/{}round_{}_ArticleLabel.csv'.format(Results_FolderName,round_num))
    
    return seg_cm, article_cm

# 平均各個 round 的segment/article score
def outputResultAvgFromRound(Results_FolderName,round_times):
    ResultsFileName = {"Segment":"SegmentScore.csv","Article":"ArticleScore.csv"}
    
    for mode in ResultsFileName:
        # 先讀 round 0
        df_results = pd.read_csv('result/{}round_{}_{}'.format(Results_FolderName,0,ResultsFileName[mode]))

        # read round 1 to 4
        for round_num in range(1,round_times):
            df_round = pd.read_csv('result/{}round_{}_{}'.format(Results_FolderName,round_num,ResultsFileName[mode]))
            df_results = df_round + df_results

        df_results = round(df_results/round_times, 5) # 四捨五入
        df_results.to_csv('result/{}_{}ScoreAverage.csv'.format(Results_FolderName,mode), index=False)
    return

# Get overall segment/article score for 5 folds 
  
def outputOverallResult(Results_FolderName,results, mode=''):
    r_sum = np.array(results[0])
    
    for r in results[1:]:
        r_sum += np.array(r)
    
    actual = np.array([0,0,0,0,0])
    predict = np.array([0,0,0,0.0,0.0])
    precision = np.array([0.0,0.0,0.0,0.0,0.0])
    recall = np.array([0.0,0.0,0.0,0.0,0.0])
    accuracy = np.array([0.0,0.0,0.0,0.0,0.0])
    f1_score = np.array([0.0,0.0,0.0,0.0,0.0])
    labels = np.array([-2,-1,1,2,0])
    
    dim = len(results)
    correct = 0  
    total = 0          
    for i in range(dim):
        count = 0
        for j in range(dim):
            count += r_sum[i][j] 
            total += r_sum[i][j]
            if (i==j):
               correct+= r_sum[i][j] 
        actual[i] = count
                      
    for j in range(dim):
        count = 0
        for i in range(dim):
            count += r_sum[i][j]
        predict[j] = count
     
    
    for i in range(dim):
        if actual[i]!=0:
           recall[i] = round(r_sum[i][i]/actual[i],4)
        else:    
           recall[i] = 0
    for j in range(dim):
        if predict[j] !=0:
           precision[j] = round(r_sum[j][j]/predict[j],4)
        else:    
           precision[j] = 0
    for i in range(dim):
        accuracy[i] = round(correct/total,4)
        if (precision[i]+recall[i])!= 0:
            f1_score[i] = round((2 * precision[i] * recall[i])/(precision[i]+recall[i]),4)
        else:
            f1_score[i] = 0
    
    out_result = []
    for i in range(dim):               
        out_result.append([labels[i],accuracy[i],precision[i],recall[i],f1_score[i]])
            
    cm_outfile = 'result/{}round_{}_cm.png'.format(Results_FolderName, mode)
    draw_confusion_matrix(r_sum, cm_outfile)
    
    df_results = pd.DataFrame(out_result, columns = ['class','accuracy','precision','recall','f1-score'])
    df_results.to_csv('result/{}_{}_Overall_Score.csv'.format(Results_FolderName,mode), index=False)
    return 


#############
# Main
#############
if __name__ == '__main__':

  FolderName = "" 
  round_times = 5 # round 次數
  
  parser = ArgumentParser()
  parser.add_argument("-r", "-- round times", dest="round_times")
  parser.add_argument("-p", "-- path of folder", dest="FolderName")
  parser.add_argument("-a", "-- aspect", dest="aspect")
  parser.add_argument("-d", "-- decision", dest="article_decision", default = "for_s")


  args = parser.parse_args()
  if (args.round_times):
     round_times = int(args.round_times)
  if (args.FolderName):
     FolderName = args.FolderName +'/'
  if (args.aspect):   
     aspect = args.aspect
  if (args.article_decision): 
     article_decision = args.article_decision
  
  seg_results = []
  article_results = []     
  for round_num in range(round_times):
      s_result, a_result = ModelPriorityClassifier(FolderName,round_num,article_decision=article_decision)
      seg_results += [s_result]
      article_results += [a_result]

  # 把五個round的結果拿來平均
  #outputResultAvgFromRound(FolderName,round_times)
  outputOverallResult(FolderName,seg_results,mode = "Segment")
  outputOverallResult(FolderName,article_results,mode = "Article")
  
  result_folder = './result/'+FolderName
  out_results(result_folder, round_times)



