#!/bin/bash


python preprocess_modify.py -a ha1;

# # # all train(expect np)

python train.py;

# # all test


python preprocess_modify.py -t test -a ha1;


# # all evaluation

python evaluation_nolabel5.py  #label list 第"5"砍掉
python ModelPriorityStrategy.py -m 5   #check_NP()控制，只要有輸入東西(非預設值)就會確認是test哪個模型(只有"np"有另外的test)


# 將stage1預測為positive，且article預測非0的文章做為test
# 輸出5model

python preprocess_modify.py -t test -m combine_article -a ha1/article -s 1 ;


python evaluation_nolabel5.py   #label list 第"5"加回來
python ModelPriorityStrategy_ha_common.py -d 'rule' 
python combineResult.py -c ha1/article





# # np model train

python preprocess_modify.py -p np -a ha1;

# # # # np model test

python preprocess_modify.py -p np -t test -a ha1;


# # # # # np train

python train.py -p np;

# # # #  # np evaluation

python evaluation.py -m np;
python ModelPriorityStrategy.py -m np;



# 輸出階層式整體結果

python preprocess_modify.py -p np -t test -a ha1;
python preprocess_modify.py -t test -a ha1;


python evaluation.py   #label list 第"5"加回來
python ModelPriorityStrategy.py 


# 將stage1預測為positive的做為test
# 輸出階層式整體結果


python preprocess_modify.py -p np -t test -m combine_article -a ha1/article;
python preprocess_modify.py -t test -m combine_article -a ha1/article;


python evaluation.py   #label list 第"5"加回來
python ModelPriorityStrategy.py 



