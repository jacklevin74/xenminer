import os
import pandas as pd
import plotly.graph_objects as go

# Get all CSV files in the current directory
folder_path = os.getcwd()
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

dfs = {f: pd.read_csv(f, comment='#') for f in csv_files}

# Specify the reference curve and difficulty
reference_curve = 'teslak80'
reference_difficulty = 90010

# Initialize figure
fig = go.Figure()

hash_speeds_at_difficulty = {}

for file, data in dfs.items():
    base_name = os.path.basename(file)
    label = base_name.rsplit('_', 1)[1].rstrip('.csv')
    
    # Store hash speed at the reference difficulty for sorting
    hash_speeds_at_difficulty[label] = data.loc[data['Difficulty'] == reference_difficulty, 'HashSpeed'].values[0]

# Sort labels
sorted_labels = sorted(hash_speeds_at_difficulty.keys(), key=lambda x: hash_speeds_at_difficulty[x], reverse=True)

# Plot the curves in the sorted order
for label in sorted_labels:
    data = dfs[[f for f in csv_files if label in f][0]]
    
    fig.add_scatter(x=data['Difficulty'], y=data['HashSpeed'], mode='lines', name=label)
    
    # Add text annotation
    if reference_difficulty in data['Difficulty'].values:
        fig.add_annotation(
            x=reference_difficulty,
            y=data.loc[data['Difficulty'] == reference_difficulty, 'HashSpeed'].values[0],
            text=f"{label}:{round(data.loc[data['Difficulty'] == reference_difficulty, 'HashSpeed'].values[0],2)}",
            showarrow=False,
        )

# Update layout
fig.update_layout(title='Benchmark Results',
                  xaxis_title='Difficulty',
                  yaxis_title='HashSpeed')

fig.show()