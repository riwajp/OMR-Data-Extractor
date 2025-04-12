import pandas as pd
import ast
import numpy as np

# Load the CSV file
csv_path = "./marked_binary/mark_positions.csv"
df = pd.read_csv(csv_path)

# Convert string to list of tuples, then filter and convert back to np.array
def filter_coords(coord_str):
    coords = np.array(ast.literal_eval(coord_str))
    filtered = [tuple(pt) for pt in coords if pt[0] <= 900]
    return np.array(filtered)

# Apply filtering and update the dataframe
df['marked'] = df['marked'].apply(filter_coords)
df['count'] = df['marked'].apply(len)

# Optional: Save to a new CSV
filtered_csv_path = "./mark_positions_filtered.csv"
df.to_csv(filtered_csv_path, index=False)

# Show result
print(df.head())
