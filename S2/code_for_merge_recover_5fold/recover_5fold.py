import os
import pandas as pd
import glob


def recover_5fold(labels_path, csv_files, updated_csv_path, drop_columns=None):

   # Load the label data from Excel
   
   labels_df = pd.read_excel(labels_path)
   # Convert the 'id' column to string type
   labels_df['id'] = labels_df['id'].astype(str)

   # Get a list of all CSV files in the specified directory
   csv_files = glob.glob(csv_files)

   # Create the directory if it doesn't exist
   os.makedirs(updated_csv_path, exist_ok=True)
    
   for csv_file in csv_files:
     try:
       # Load the current CSV file
       data_df = pd.read_csv(csv_file)
    
       # Iterate over each row in the CSV file
       for index, row in data_df.iterrows():
           if str(row['S2']) != '0':
               # Find the corresponding row in the label data by 'id'
               label_row = labels_df.loc[labels_df['id'] == str(row['id'])]
               # If a corresponding row is found, update 'S2' and '態度分數句'
               if not label_row.empty:
                   data_df.at[index, 'S2'] = label_row['S2'].values[0]
                   data_df.at[index, '態度分數句'] = label_row['態度分數句'].values[0]
                         
           
     except Exception as e:
       print(f"Error processing file {csv_file}: {e}")
            
     # Save the updated CSV file
     # Define the path for the updated file
     updated_csv_file = os.path.join(updated_csv_path, f'{os.path.basename(csv_file)}')
     data_df.to_csv(updated_csv_file, index=False)

   return
    
#############
# Main
#############
if __name__ == '__main__': 
   recover_5fold('data_for_merge/train_label0326.xlsx','data_for_merge/train_5fold_data/*.csv', 'updated_data_for_merge/train_5fold_data/')
   recover_5fold('data_for_merge/train_label0326.xlsx','data_for_merge/test_5fold_data/*.csv', 'updated_data_for_merge/test_5fold_data/')
   
