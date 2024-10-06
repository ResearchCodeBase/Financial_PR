# coding=utf-8
import os
import numpy as np
import pandas as pd

# Read the sheet named "Austria_2020" from the Excel file
sheet_name = "Austria"
year = 2020
df = pd.read_excel('foreign_data.xlsx', sheet_name=sheet_name)

# Calculate external assets, interbank assets, and interbank liabilities
external_assets = df['2020Cash and Balances with Central Banks'] + df['2020_Total Assets'] - df['2020_Total Liabilities']
Interbank_Assets = df['2020Net Loans to Banks']
Interbank_Liabilities = df['2020Total Deposits from Banks']

length = len(external_assets)

# Convert to DataFrame and transpose
external_assets_df = pd.DataFrame(external_assets).T
Interbank_Assets_df = pd.DataFrame(Interbank_Assets).T
Interbank_Liabilities_df = pd.DataFrame(Interbank_Liabilities).T

print('External assets length:', length)
print(external_assets)
print('Interbank assets length:', len(Interbank_Assets))
print('Interbank liabilities length:', len(Interbank_Liabilities))

# Create subdirectory for saving data
dir_path = f'Europe/{sheet_name}/Original_Data'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Save as different txt files, displaying data in rows
external_assets_file_path = f'{dir_path}/External_Assets_No_Other_Banks_{year}.txt'
external_assets_df.to_csv(external_assets_file_path, sep='\t', index=False, header=False)

# Append a '0' at the end of the row vector
external_assets_df[len(external_assets_df.columns)] = 0

# Save another file with an appended '0'
external_assets_with_zero_file_path = f'{dir_path}/External_Assets_With_Other_Banks_{year}.txt'
external_assets_df.to_csv(external_assets_with_zero_file_path, sep='\t', index=False, header=False)

Interbank_Assets_df.to_csv(f'{dir_path}/Interbank_Assets_{year}.txt', sep='\t', index=False, header=False)
Interbank_Liabilities_df.to_csv(f'{dir_path}/Interbank_Liabilities_{year}.txt', sep='\t', index=False, header=False)

# Rename 'Name' column to 'Country'
bank_names = df['Name'].rename('Country')

# Define the file path for saving bank names
bank_names_file_path = f'{dir_path}/Bank_Names_{year}.csv'

# Save as a CSV file with the column name 'Country'
bank_names.to_csv(bank_names_file_path, index=False)

# Save the bank count
length_file_path = f'{dir_path}/Bank_Count.txt'

# Save the integer data to a text file
with open(length_file_path, 'w') as file:
    file.write(str(length))

e1 = np.loadtxt(external_assets_file_path)

print('Loaded external assets:', len(e1))
print(e1)
print(f'{year} Bank names saved successfully')
print(f'{year} Data saved successfully')
