import pandas as pd
import glob as glob

# File paths of CSV files to merge
file_paths = glob.glob('data_for_merge/train_5fold_data/*.csv')

# Read all CSV files and concatenate them into a single DataFrame
dfs = [pd.read_csv(file) for file in file_paths]
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicate rows based on the 'id' columns
merged_df.drop_duplicates(subset=['id'], inplace=True)

# Write the merged and de-duplicated DataFrame to a new CSV file
merged_df.to_csv('data_for_merge/merged_train.csv', index=False)

# File paths of CSV files to merge
file_paths = glob.glob('data_for_merge/test_5fold_data/*.csv')

# Read all CSV files and concatenate them into a single DataFrame
dfs = [pd.read_csv(file) for file in file_paths]
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicate rows based on the 'id' columns
merged_df.drop_duplicates(subset=['id'], inplace=True)

# Write the merged and de-duplicated DataFrame to a new CSV file
merged_df.to_csv('data_for_merge/merged_test.csv', index=False)
