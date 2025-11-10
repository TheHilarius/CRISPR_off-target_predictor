#/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

train_data =torch.load("data/train_embeddings.pt")
val_data = torch.load("data/val_embeddings.pt")
test_data = torch.load("data/test_embeddings.pt")

# print(f"Train data keys: {train_data.keys()}")
# print(f"Validation data keys: {val_data.keys()}")   
# print(f"Test data keys: {test_data.keys()}")

# print(f"\nTrain input_ids shape: {train_data['input_ids'].shape}")
# print(f"Train delta shape: {train_data['delta'].shape}")
# print(f"Train labels shape: {train_data['labels'].shape}")

# print(f"\nVal input_ids shape: {val_data['input_ids'].shape}")
# print(f"Val labels shape: {val_data['labels'].shape}")

# print(f"\nTest input_ids shape: {test_data['input_ids'].shape}")
# print(f"Test labels shape: {test_data['labels'].shape}")

class CrossSeqTransformer(nn.Module):
    def __init__(self, vocab_size=6, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 max_len=47, dropout=0.1, dg_embedding_dim=16):
        super().__init__()
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model) # positional encoding
        
        # Transformer (encoder-decoder)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead, # multihead attention
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        # delta G embedding
        self.dg_embed = nn.Linear(1, dg_embedding_dim)

        # Output regression head
        combined_dim = d_model + dg_embedding_dim
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, delta_g):
        B, L = input_ids.shape
        device = input_ids.device

        # Positional encoding
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B,-1) # position indices

        # Embed sequences for encoder-decoder architecture
        seq_embed = self.token_embed(input_ids) + self.pos_embed(pos)
        on_target = seq_embed[:, :23, :]
        off_target = seq_embed[:, 24:, :]

        # Transformer expects (batch, seq, dim)
        out = self.transformer(on_target, off_target)  # shape [B, L, d_model]

        # Mean pooling sequence length
        pooled = out.mean(dim=1)
        
        # delta G embedding
        dg_emb = self.dg_embed(delta_g)

        # Concatenate all features
        combined = torch.cat([pooled, dg_emb], dim=1)

        # Predict activity
        activity_pred = self.regressor(combined)

        return activity_pred.squeeze(-1)

## -------- Training Loop --------- ##

model = CrossSeqTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## NOTE: 10 epochs and batches for now
num_epochs = 10
epoch_loss = 0
train_ids = train_data['input_ids']
train_delta = train_data['delta'].unsqueeze(-1) if train_data['delta'].dim() == 1 else train_data['delta']
train_labels = train_data['labels'].unsqueeze(-1) if train_data['labels'].dim() == 1 else train_data['labels']

train_dataset = TensorDataset(train_ids, train_delta, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_delta_min = train_data['delta'].min()
train_delta_max = train_data['delta'].max()
train_delta_norm = (train_delta - train_delta_min) / (train_delta_max - train_delta_min)

val_ids = val_data['input_ids']
val_delta = val_data['delta'].unsqueeze(-1) if val_data['delta'].dim() == 1 else val_data['delta']
val_delta_norm = (val_delta - train_delta_min) / (train_delta_max - train_delta_min)
val_labels = val_data['labels'].unsqueeze(-1) if val_data['labels'].dim() == 1 else val_data['labels']

val_dataset = TensorDataset(val_ids, val_delta_norm, val_labels)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_ids = test_data['input_ids']
test_delta = test_data['delta'].unsqueeze(-1) if test_data['delta'].dim() == 1 else test_data['delta']
test_delta_norm = (test_delta - train_delta_min) / (train_delta_max - train_delta_min)
test_labels = test_data['labels'].unsqueeze(-1) if test_data['labels'].dim() == 1 else test_data['labels']

test_dataset = TensorDataset(test_ids, test_delta_norm, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_dataset = TensorDataset(train_ids, train_delta_norm, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for input_ids, delta_g, labels in train_loader:
        input_ids = input_ids.to(device)
        delta_g = delta_g.to(device)
        labels = labels.to(device)
        
        predictions = model(input_ids, delta_g)
        loss = criterion(predictions, labels.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    print(f"Epoch {epoch+1}: avg loss = {epoch_loss / num_batches:.4f}")

print("Training complete")

model.eval()
test_loss = 0.0
num_test_batches = 0
predictions_all = []
actuals_all = []

with torch.no_grad():
    for input_ids, delta_g, labels in test_loader:
        input_ids = input_ids.to(device)
        delta_g = delta_g.to(device)
        labels = labels.to(device)
        
        predictions = model(input_ids, delta_g)
        loss = criterion(predictions, labels.squeeze(-1))
        
        test_loss += loss.item()
        num_test_batches += 1
        
        predictions_all.append(predictions.cpu().numpy())
        actuals_all.append(labels.cpu().numpy())

predictions_all = np.concatenate(predictions_all)
actuals_all = np.concatenate(actuals_all)

print(f"\nTest Loss: {test_loss / num_test_batches:.4f}\n")
print("Sample Predictions vs Actual:")
for i in range(min(20, len(predictions_all))):
    print(f"  Predicted: {predictions_all[i][0]:.2f}, Actual: {actuals_all[i][0]:.2f}")