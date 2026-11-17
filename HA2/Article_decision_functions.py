
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
            if lab_counts[-1][1] > lab_counts[1][1]: # -1 appears after 1
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
               if lab_counts[-2][0] > 0:
                  Label = -1
               else:   
                  Label = 1   
            else:   
               Label = 0          
    return Label          
        

