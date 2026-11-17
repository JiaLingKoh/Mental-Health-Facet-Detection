import os
import pandas as pd
import glob

drop_columns = ['Unnamed: 0', 'stage1Sentence', '態度分數句_未剪短', '內文'] 

def update_test_files(fpath):
    train_df = pd.read_csv(fpath+'merged_train.csv')
    test_df = pd.read_csv(fpath+'merged_test.csv')
    
    test_df['id'] = test_df['id'].astype(str)
    train_df['id'] = train_df['id'].astype(str)

    # Merge with suffixes to distinguish columns from the test and train DataFrames
    t_merged_df = pd.merge(test_df, train_df, on='id', how='left', suffixes=('', '_new'))
    
    # Update the old DataFrame with the new data where available
    for col in test_df.columns:
        if col != 'id' and col in train_df.columns:
            # Use the new values from t_merged_df where available
            t_merged_df[col] = t_merged_df[col + '_new'].combine_first(t_merged_df[col])

    # Keep only the original columns from the test DataFrame
    t_merged_df = t_merged_df[test_df.columns]

    # Save the updated DataFrame to a new CSV file
    t_merged_df.to_csv(fpath+'merged_test.csv', index=False)
    return
    
    
def recover_5fold(merged_file, old_csv_path, updated_csv_path): 
  # Load the new CSV file into a DataFrame
  new_df = pd.read_csv(merged_file)
  new_df['id'] = new_df['id'].astype(str)

  # File paths of old CSV files to update
  file_paths = glob.glob(old_csv_path)

  # Directory path you want to create
  directory_path = updated_csv_path

  # Create the directory if it doesn't exist
  os.makedirs(directory_path, exist_ok=True)

  for file in file_paths:
    # Read the old file
    old_df = pd.read_csv(file)
    
    # Drop the specified columns
    for col in drop_columns:
        if col in old_df.columns:
           old_df = old_df.drop(columns=[col])
     
    old_df['id'] = old_df['id'].astype(str)

    # Change the merge to 'inner' to only keep rows present in both DataFrames
    merged_df = pd.merge(old_df, new_df, on='id', how='inner', suffixes=('', '_new'))
    
    # Update the old DataFrame with the new data where available
    for col in new_df.columns:
        if col != 'id' and col in old_df.columns:
            # Use the new values from merged_df where available
            merged_df[col] = merged_df[col + '_new'].combine_first(merged_df[col])

    # Keep only the original columns from the old DataFrame
    merged_df = merged_df[old_df.columns]

    # Save the updated DataFrame to a new CSV file in the specified directory
    new_file_name = os.path.join(directory_path, os.path.basename(file))
    merged_df.to_csv(new_file_name, index=False)

  return


#############
# Main
#############
if __name__ == '__main__': 
   update_test_files('data_for_merge/')
   recover_5fold('data_for_merge/merged_train.csv','data_for_merge/train_5fold_data/*.csv', 'updated_data_for_merge/train_5fold_data/')
   recover_5fold('data_for_merge/merged_test.csv','data_for_merge/test_5fold_data/*.csv', 'updated_data_for_merge/test_5fold_data/')
   
