import pandas as pd

# Define variables
country_to_filter = 'Italy'
year_to_filter = '2020'

# Load Excel file
df = pd.read_excel('foreign_data.xlsx')  # Replace 'foreign_data.xlsx' with your file name

# Handle any whitespace in column names
df.columns = df.columns.str.strip()

# Filter data by Country column
filtered_df = df[df['Country'] == country_to_filter]

# Select specific columns, ensuring there are no spaces around column names
selected_columns = [
    'Name',
    'Country',
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
]

# Create a DataFrame copy with the selected columns
intermediate_df = filtered_df[selected_columns].copy()

# Drop rows with NA values, checking specific columns
final_df = intermediate_df.dropna(subset=[
    f'{year_to_filter}_Total Assets',
    f'{year_to_filter}_Total Liabilities',
    f'{year_to_filter}Cash and Balances with Central Banks',
    f'{year_to_filter}Net Loans to Banks',
    f'{year_to_filter}Total Deposits from Banks'
])

# Build the name for the output sheet
sheet_name = f'{country_to_filter}_{year_to_filter}'

# Save the results to a new sheet in foreign_data.xlsx
with pd.ExcelWriter('foreign_data.xlsx', mode='a', engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name=sheet_name, index=False)
