import pandas as pd

# Load the CSV file
df = pd.read_csv('data/historical_data.csv')

# List of columns to delete
columns_to_delete = ['unix', 'tradecount']

# Delete the columns
df.drop(columns=columns_to_delete, inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv('data/historical_data_modified.csv', index=False)