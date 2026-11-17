import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn.metrics import cohen_kappa_score

import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser


round_times =5
current_path_stage1 = ''
current_path_stage2 = 'result/'
save_file_name = 'result/_Article_all_result'


def cm_plot4(y_true,y_pred):
    array = np.array([y_true,y_pred]).T

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[-2, -1, 1, 2, 0])
    
        # 将混淆矩阵转换为DataFrame
    # conf_df = pd.DataFrame(conf_matrix, index=[-2,-1,1,2], columns=[-2,-1,1,2])
    conf_df = pd.DataFrame(conf_matrix, index=[-2, -1, 1, 2, 0], columns=[-2, -1, 1, 2, 0])
    # conf_df.to_csv(f'{output_dir}/fold_{fold}_confusion_matrix.csv' ,index= True)
    # 使用Seaborn绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 保存为JPG图像
    # plt.savefig(f'{output_dir}/fold_{fold}_confusion_matrix.jpg', format='jpg', dpi=300)

    # plt.show()
    plt.savefig(f'{save_file_name}_cm.png')

    
    
    

parser = ArgumentParser()
parser.add_argument("-c", "-- current path stage1", dest="current_path_stage1")

args = parser.parse_args()
if (args.current_path_stage1):
   current_path_stage1 = args.current_path_stage1

for i in range(round_times):    
    temp_stage1 = pd.read_csv(current_path_stage1+'/prediction{}.csv'.format(i)) # read csv file
    temp_stage2 = pd.read_csv(current_path_stage2+'round_{}_ArticleLabel.csv'.format(i)) # read csv file
    df_temp = pd.merge(temp_stage1, temp_stage2, left_on="id", right_on="id",how="outer")
    if i == 0:
        df_all = df_temp
    else:
        df_all = pd.concat([df_all,df_temp],axis=0)

# print(df_all)
df_all.rename(columns = {'model_pred':'stage2_article_pred', 'ModelPred_articleLabel':'stage2_segment_pred'}, inplace = True)
df_all = df_all.drop(['Unnamed: 0', 'article_label', 'label'], axis=1)
df_all.to_csv('result/all_results/Article_all_result.csv', encoding='utf-8_sig')


df_all = df_all.fillna(0)

cm_plot4(df_all['S1'],df_all['stage2_segment_pred'])

label = [-2, -1, 1, 2, 0]
# print(classification_report(df_all['stage2SCORE'],df_all['ModelPred_articleLabel'],labels=label, digits =4))
df_result = pd.DataFrame(classification_report(df_all['S1'],df_all['stage2_segment_pred'],labels=label, digits =4, output_dict=True)).T
df_result = df_result.round(4)
df_result.to_csv(f'{save_file_name}_Score.csv', encoding='utf-8_sig')
