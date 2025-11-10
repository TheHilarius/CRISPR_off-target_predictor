import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
np.random.seed(0)

# Activity between 0 and 1


### Linear transformation

# Depending on the transformer outpt. In this case I put dummy variables, of 4 and 768? ###  To be changed
batch_size = 4 
latent_dim = 768
transformer_output = torch.randn(batch_size, latent_dim)

# Define linear layer (fully connected). Out feature is just activity, so dimension is 1.
activity_head = nn.Linear(in_features=latent_dim, out_features=1)

# Run the NN
predicted_activity = activity_head(transformer_output)

print("predicted activity shape: ", predicted_activity.shape)


# Could also look something like this, if we have a classs structure # I think we do.
class OffTargetPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        
        return self.fc(x)

model = OffTargetPredictor(input_dim=768)
x = torch.randn(10, 768)  # e.g., transformer outputs for 10 off-targets
y_pred = model(x)
print("Predicted y_pred.shape: ", y_pred.shape)  # torch.Size([10, 1])


### Ranking

# loading data 
path = r"data/filtered_change_seq.csv" #Relative path
df = pd.read_csv(path)

# Treat CHANGEseq_reads as 'experimental activity'
df['experimental_activity'] = df['CHANGEseq_reads']

# Simulate predicted activity (for now we can add a bit of random noise)
df['predicted_activity'] = df['CHANGEseq_reads'] + np.random.normal(0, 50, size=len(df))

# Rank both experimental and predicted 
df['exp_rank'] = df['experimental_activity'].rank(ascending=False)
df['pred_rank'] = df['predicted_activity'].rank(ascending=False)

# Compare rankings with spearman correlation
rho, pval = spearmanr(df['experimental_activity'], df['predicted_activity'])
print(f"Spearman correlation between experimental and predicted activity: {rho:.3f} (p={pval:.3e})")

# Sort by predicted activity and display
ranked = df.sort_values('predicted_activity', ascending=False)
print("\nTop off-targets by predicted activity:\n")
print(ranked[['name', 'offtarget_sequence', 'predicted_activity', 'experimental_activity']])

