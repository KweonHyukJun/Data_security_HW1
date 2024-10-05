import pandas as pd

# Define the file path to the uploaded CSV file
file_path = './spam.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Display the contents of the CSV file
print("Loaded CSV Data:")
print(data.head())  # Display the first 5 rows of the dataframe

# Display basic information about the DataFrame
print("\nData Information:")
print(data.info())

# Display the column names and first few rows
print("\nColumn Names and Sample Data:")
print(data.columns)
print(data.sample(5))  # Display 5 random rows
