# # Data preprocessing main

from argparse import ArgumentParser
# import custom library
from lib.preprocess import *

# process data

round_times = 5
aspect = ''
np_model = '5_model'
test_model = 'train'

#資料sample 0分segment數量

label_dic = {'-2':0,'-1':1,'1':2,'2':3,'0':4,'5':5, '-5':6 }
negative_sample_times = {'-2':5,'-1':5,'1':5,'2':5,'0':3, '5':5, '-5':1}

gen_type = 'train'


#===============================================================================                   
parser = ArgumentParser()
parser.add_argument("-r", "-- round times", dest="round_times")
parser.add_argument("-t", "-- data type train/test", dest="gen_type")
parser.add_argument("-a", "-- aspect", dest="aspect")
parser.add_argument("-l", "-- minnum segment length", dest="min_segLen")
parser.add_argument("-p", "-- is np model?", dest="np_model")
parser.add_argument("-m", "-- test model?", dest="test_model")
parser.add_argument("-s", "-- perform given stage", dest="stage_num")

args = parser.parse_args()
if (args.round_times):
   round_times = int(args.round_times)
if (args.gen_type):
   gen_type = args.gen_type
if (args.aspect):   
   aspect = args.aspect
if (args.np_model):   
   np_model = args.np_model
if (args.min_segLen):   
   min_segLen = int(args.min_segLen)
if (args.test_model):   
   test_model = args.test_model
if (args.stage_num):   
   stage_num = args.stage_num

if np_model != 'np':
  if stage_num == '1': 
   # perform stage 1
     label_list = ['0'] 
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 1800, exclusive = False, continous_answer = True, np_model = np_model, test_model = test_model)  
 
     label_list = ['2']
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 0, exclusive = True,continous_answer = True, np_model = np_model, test_model = test_model)

     label_list = ['-2']
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 0, exclusive = True,continous_answer = True, np_model = np_model, test_model = test_model)

     label_list = ['1']
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 0, exclusive = True,continous_answer = True, np_model = np_model, test_model = test_model)

     label_list = ['-1']
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 0, exclusive = True,continous_answer = True, np_model = np_model, test_model = test_model)

  elif stage_num == '2':  
   # perform stage 2
     label_list = ['-1','-2','0','1','2']
     preprocess_stage2(label_list, gen_type, round_times, negative_sample_times = negative_sample_times) 
 
elif np_model == 'np':
  np = '_np'
  if stage_num == '1': 
  # perform stage 1
     label_list = ['0'] 
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 1800, exclusive = False,continous_answer = True, np_model = np_model, test_model = test_model)  
     label_list = ['5']
     preprocess_stage1(label_list, round_times, gen_type, aspect, sample_0_number = 0, exclusive = True,continous_answer = True, np_model = np_model, test_model = test_model)
   
  elif stage_num == '2':  
     # perform stage 2
     label_list = ['0','5']
     preprocess_stage2(label_list, gen_type, round_times, negative_sample_times = negative_sample_times)  
   
