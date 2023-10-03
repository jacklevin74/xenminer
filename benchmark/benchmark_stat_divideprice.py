import os
import pandas as pd
import plotly.graph_objects as go

# Get all CSV files in the current directory
folder_path = os.getcwd()
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

dfs = {f: pd.read_csv(f, comment='#') for f in csv_files}

price_file = os.path.join(folder_path, 'prices.txt')
prices = {}
if os.path.exists(price_file):
    with open(price_file, 'r') as f:
        for line in f:
            label, price = line.strip().split(':')
            prices[label] = float(price)
else:
    for f in csv_files:
        label = os.path.basename(f).rsplit('_', 1)[1].rstrip('.csv')
        prices[label] = 1.0
    with open(price_file, 'w') as f:
        for label, price in prices.items():
            f.write(f"{label}:{price}\n")

# Specify the reference curve and difficulty
reference_curve = 'teslak80'
reference_difficulty = 90010


# Initialize figure
fig = go.Figure()

# Step 1: Adjust hash speed by price and find reference values
reference_values = None
for file, data in dfs.items():
    base_name = os.path.basename(file)
    label = base_name.rsplit('_', 1)[1].rstrip('.csv')
    data['AdjustedHashSpeed'] = data['HashSpeed'] / prices[label]
    
    if label == reference_curve:
        reference_values = data['AdjustedHashSpeed'].values

# Step 2: Normalize by reference values and plot
hash_speeds_at_difficulty = {}

for file, data in dfs.items():
    base_name = os.path.basename(file)
    label = base_name.rsplit('_', 1)[1].rstrip('.csv')
    
    # Normalize
    data['NormalizedHashSpeed'] = data['AdjustedHashSpeed'] / reference_values
    
    # Store hash speed at the reference difficulty for sorting
    hash_speeds_at_difficulty[label] = data.loc[data['Difficulty'] == reference_difficulty, 'NormalizedHashSpeed'].values[0]

# Sort labels
sorted_labels = sorted(hash_speeds_at_difficulty.keys(), key=lambda x: hash_speeds_at_difficulty[x], reverse=True)

# Plot the curves in the sorted order
for label in sorted_labels:
    data = dfs[[f for f in csv_files if label in f][0]]
    
    fig.add_scatter(x=data['Difficulty'], y=data['NormalizedHashSpeed'], mode='lines', name=label)
    
    # Add text annotation
    if reference_difficulty in data['Difficulty'].values:
        fig.add_annotation(
            x=reference_difficulty,
            y=data.loc[data['Difficulty'] == reference_difficulty, 'NormalizedHashSpeed'].values[0],
            text=f"{label}:{round(data.loc[data['Difficulty'] == reference_difficulty, 'NormalizedHashSpeed'].values[0],2)}",
            showarrow=False,
        )

# Update layout
fig.update_layout(title='Benchmark Results Normalized by Price and Reference Curve',
                  xaxis_title='Difficulty',
                  yaxis_title='Normalized HashSpeed')

fig.show()