
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
       if test_value/compare_value > 2:          
          return True
       else:
          return False
          
# decide article label
def det_article_label(p_id, seg_labels, article_predict, NP_predict, article_decision):  
    lab_counts = {-2:[0,-1,-1], -1:[0,-1,-1], 1:[0,-1,-1], 2:[0,-1,-1],-3:[0,-1,-1]} # appearing counts, last appear loc, first appear loc
    non_zero = False    
    Label = 0
            
    for i, l in enumerate(seg_labels):             
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
                   if lab_counts[l][2]== -1:
                      lab_counts[l][2]= i  
                                   
    if not non_zero:
        #Label = article_predict
        #Label = 0
        
        if (NP_predict > 0 and article_predict > 0) or (NP_predict < 0 and article_predict < 0) or NP_predict == 0:
           if article_predict == 1:
              Label = 0
           else:
              Label = article_predict   
        else:
           Label = 0       
           
        # use article_predict to 決定PO文的label (all are 0s)                
    elif article_decision == 'voting':
        Label = max(lab_counts, key= lambda key: lab_counts[key][0])  # the label with the maximum count 
    elif article_decision == 'v&last':
        Label = max(lab_counts, key= lambda key: lab_counts[key])      
        if lab_counts[2][0] > 0:    
            Label = 2
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
        n_count = lab_counts[-2][0] + lab_counts[-1][0]
        v_Label = max(lab_counts, key= lambda key: lab_counts[key])  # vote & last
        l_label = max(lab_counts, key= lambda key: lab_counts[key][1])  # the label with the last position
        
        if (v_Label == article_predict and article_predict < 0): # Article predicts negative and consistent with v_label
           Label = v_Label
            
        # Rules of deciding -1
        elif (lab_counts[-1][0] > 0 and lab_counts[-2][0] > 0) and p_count == 0: # detect -1 and -2 and no positives
             if majorty(lab_counts[-2][0],lab_counts[-1][0]):  # -2 >> -1
                Label = -2
             else:
                Label = -1
             if article_predict < 0 and Label != article_predict:
                   Label = article_predict              
        elif (lab_counts[-1][0] > 0 and p_count == 0):  # only detect -1 and no positives   
              Label = -1  
        elif (NP_predict == -5 and lab_counts[-2][0] > 0 and p_count == 0):
              Label = -2     
        # Rules of deciding 2      
        elif (lab_counts[2][0] > 0): # detect 2 and no negatives
             if (lab_counts[-2][0] > 0 and lab_counts[-1][0] > 0 and lab_counts[2][0] < 2):
                Label = 1
             else:   
                Label = 2
                
        # Rules of deciding 1           
        elif (v_Label == 1 and n_count == 0 and article_predict > 0 and NP_predict != 0):  
             Label = 1               
        elif (v_Label == 1 and n_count == 0 and NP_predict > 0):  
             Label = 1      
        elif (v_Label == 1 and n_count != 0 and majorty(lab_counts[1][0],n_count)):  # 1 >> negatives
             Label = 1
        else:
             Label = 0     
        '''       
        elif NP_predict == -5:  
           if v_Label == -1: 
              Label = -1
           #elif p_count > 0:
           #   Label = -1
           else:
              Label = -2      
        elif NP_predict == 5:
           if v_Label == 1: 
              Label = 1
           #elif p_count > 0:
           #   Label = -1
           else:
              Label = 2      
        elif NP_predict == 0: 
             Label = article_predict       
        '''     
        
    return Label          
        

