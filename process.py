import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('response.csv')
df = df[df['Round'] == 5]
# Filter columns based on their names
filtered_columns = [col for col in df.columns if col.endswith("precision")]
filtered_df = df[filtered_columns]
df['precision'] = filtered_df.mean(axis=1)
df = df.drop(columns=filtered_columns)

filtered_columns = [col for col in df.columns if col.endswith("recall")]
filtered_df = df[filtered_columns]
df['recall'] = filtered_df.mean(axis=1)
df = df.drop(columns=filtered_columns)

filtered_columns = [col for col in df.columns if col.endswith("f1-score")]
filtered_df = df[filtered_columns]
df['f1-score'] = filtered_df.mean(axis=1)
df = df.drop(columns=filtered_columns)

filtered_columns = [col for col in df.columns if col.endswith("support")]
filtered_df = df[filtered_columns]
df = df.drop(columns=filtered_columns)

# Save the DataFrame with row-wise averages to a new CSV file
df.to_csv('response_processed.csv', index=False)
