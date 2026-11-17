import pandas as pd
import os

def out_results(current_path, round_times):
  if not os.path.exists(current_path+'all_results'):
      os.makedirs(current_path+'all_results')

  # define variable
  df_wrong_all_out = []
  df_correct_all_out = [] 

  # find wrong segment and article
  """
  從ArticleLabel.csv找出錯誤的case,
  根據文章ID去SegmentLabel抓出該篇PO文的句組
  """
  for i in range(round_times):
     df = pd.read_csv(current_path+'round_{}_ArticleLabel.csv'.format(i)) # read csv file


     """篩選正確和錯誤的資料"""
     df_wrong_article = df[df['article_label'] != df['ModelPred_articleLabel']]  # 找判斷錯誤的文章ID，true label != pred_label
     df_correct_article = df[df['article_label'] == df['ModelPred_articleLabel']]    # 找判斷正確的文章ID，true label == pred_label
     # print(df_wrong_article['article_label'].count())
     # print(df_correct_article['article_label'].count())

     """convert the dataframe to list"""
     df_wrong_article_ID_list = df_wrong_article.id.to_list()
     df_correct_article_ID_list = df_correct_article.id.to_list()

     """read SegmentLabel file，用文章ID來找該篇文章所切分的句組"""
     df_segmentLabel = pd.read_csv(current_path+'round_{}_SegmentLabel.csv'.format(i))

     """rename column for segment file"""
     df_segmentLabel.rename(columns={'label':'SegmentLable'},inplace = True)

     """從 df_SegmentLabel裡面取出某幾個columns，且先用 df_(wrong/correct)_article_ID_list[0] 當作第一個比對資料。"""
     # 這行的目的是先建立好一個dataframe
     tmp_wrong_example = df_segmentLabel.loc[df_segmentLabel['id'] == df_wrong_article_ID_list[0]].loc[:,['id','segment','SegmentLable','ModelPred_SegmentLabel']]  # create tmp_wrong_example by df_wrong_article_ID_list[0]
     tmp_correct_example = df_segmentLabel.loc[df_segmentLabel['id'] == df_correct_article_ID_list[0]].loc[:,['id','segment','SegmentLable','ModelPred_SegmentLabel']]  # create tmp_correct_example by df_correct_article_ID_list[0]

     # 增加article_label的欄位
     tmp_wrong_example.insert(tmp_wrong_example.shape[1], 'ArticleLabel', int(df_wrong_article[df_wrong_article['id']==df_wrong_article_ID_list[0]]['article_label'].iloc[0]))
     tmp_correct_example.insert(tmp_correct_example.shape[1], 'ArticleLabel', int(df_correct_article[df_correct_article['id']==df_correct_article_ID_list[0]]['article_label'].iloc[0]))

     # 增加model predict article_label的欄位
     tmp_wrong_example.insert(tmp_wrong_example.shape[1], 'ModelPredArticleLabel', int(df_wrong_article[df_wrong_article['id']==df_wrong_article_ID_list[0]]['ModelPred_articleLabel'].iloc[0]))
     tmp_correct_example.insert(tmp_correct_example.shape[1], 'ModelPredArticleLabel', int(df_correct_article[df_correct_article['id']==df_correct_article_ID_list[0]]['ModelPred_articleLabel'].iloc[0]))

        

     # 把錯誤的文章ID列表尋訪一次
     for num_wrong_article_ID in range(1,len(df_wrong_article_ID_list)):

        # segment label的文章ID == 錯誤文章的ID , for ArticleLabel欄位
        tmp = df_segmentLabel.loc[df_segmentLabel['id'] == df_wrong_article_ID_list[num_wrong_article_ID]].loc[:,['id','segment','SegmentLable','ModelPred_SegmentLabel']]
        tmp.insert(tmp.shape[1], 'ArticleLabel', int(df_wrong_article[df_wrong_article['id']==df_wrong_article_ID_list[num_wrong_article_ID]]['article_label'].iloc[0]))
        
        tmp.insert(tmp.shape[1], 'ModelPredArticleLabel', int(df_wrong_article[df_wrong_article['id']==df_wrong_article_ID_list[num_wrong_article_ID]]['ModelPred_articleLabel'].iloc[0]))
        tmp_wrong_example = pd.concat([tmp_wrong_example,tmp])
            

     # 把正確的文章ID列表尋訪一次
     for num_correct_article_ID in range(1,len(df_correct_article_ID_list)):

        tmp = df_segmentLabel.loc[df_segmentLabel['id'] == df_correct_article_ID_list[num_correct_article_ID]].loc[:,['id','segment','SegmentLable','ModelPred_SegmentLabel']]
        tmp.insert(tmp.shape[1], 'ArticleLabel', int(df_correct_article[df_correct_article['id']==df_correct_article_ID_list[num_correct_article_ID]]['article_label'].iloc[0]))
        tmp.insert(tmp.shape[1], 'ModelPredArticleLabel', int(df_correct_article[df_correct_article['id']==df_correct_article_ID_list[num_correct_article_ID]]['ModelPred_articleLabel'].iloc[0]))
        
        tmp_correct_example = pd.concat([tmp_correct_example,tmp])

        
     # concat data
     if len(df_wrong_all_out) == 0:
         df_wrong_all_out = tmp_wrong_example
     else:
         df_wrong_all_out = pd.concat([df_wrong_all_out,tmp_wrong_example])

     if len(df_correct_all_out) == 0:
         df_correct_all_out = tmp_correct_example
     else:
         df_correct_all_out = pd.concat([df_correct_all_out,tmp_correct_example])


  # output file
  df_wrong_all_out.to_csv(current_path+'all_results/wrong_examples.csv',encoding='utf-8_sig',index=False)
  df_correct_all_out.to_csv(current_path+'all_results/correct_examples.csv',encoding='utf-8_sig',index=False)
  return


