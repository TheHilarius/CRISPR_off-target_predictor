#/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import os
from sklearn.model_selection import KFold
import copy

train_data =torch.load("data/train_embeddings.pt")
val_data = torch.load("data/val_embeddings.pt")
test_data = torch.load("data/test_embeddings.pt")


# print(f"Train data keys: {train_data.keys()}")
# print(f"Validation data keys: {val_data.keys()}")   
# print(f"Test data keys: {test_data.keys()}")

# print(f"\nTrain sequence_embs shape: {train_data['sequence_embs'].shape}")
# print(f"Train deltaGH shape: {train_data['deltaGH'].shape}")
# print(f"Train activity shape: {train_data['activity'].shape}")

# print(f"\nVal sequence_embs shape: {val_data['sequence_embs'].shape}")
# print(f"Val activity shape: {val_data['activity'].shape}")

# print(f"\nTest sequence_embs shape: {test_data['sequence_embs'].shape}")
# print(f"Test activity shape: {test_data['activity'].shape}")

# global flag for advanced or simple regressor
use_advanced_regressor = True

    
class CrossSeqTransformer(nn.Module):
    def __init__(self, vocab_size=6, d_model=128, nhead=8,          # WHY SHOULD VOCAB SIZE BE 6??
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
         
         # ---------------------------
         # Transformer Encoder
         # ---------------------------
         encoder_layer = nn.TransformerEncoderLayer(
             d_model=d_model,
             nhead=nhead,
             dim_feedforward=4 * d_model,
             dropout=dropout,
             batch_first=True
         )
         self.encoder = nn.TransformerEncoder(
             encoder_layer,
             num_layers=num_encoder_layers
         )

         # ---------------------------
         # Bottleneck MLP (between encoder and decoder)
         # ---------------------------
         self.mlp_bottleneck = nn.Sequential(
             nn.Linear(d_model, 512),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(512, d_model),   # final dimension returned to decoder
             nn.GELU(),
         )

         # ---------------------------
         # Transformer Decoder
         # ---------------------------
         decoder_layer = nn.TransformerDecoderLayer(
             d_model=d_model,
             nhead=nhead,
             dim_feedforward=4 * d_model,
             dropout=dropout,
             batch_first=True
         )
         self.decoder = nn.TransformerDecoder(
             decoder_layer,
             num_layers=num_decoder_layers
         )

         # ---------------------------
         # deltaGH embedding
         # ---------------------------
         self.dg_embed = nn.Linear(1, dg_embedding_dim)

         # ---------------------------
         # Regression head
         # ---------------------------
         combined_dim = d_model + dg_embedding_dim

         if use_advanced_regressor:
             self.regressor = nn.Sequential(
                 nn.Linear(combined_dim, 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(),

                 nn.Linear(512, 256),
                 nn.BatchNorm1d(256),
                 nn.GELU(),

                 nn.Linear(256, 128),
                 nn.ReLU(),

                 nn.Linear(128, 64),
                 nn.GELU(),

                 nn.Linear(64, 1),
             )
         else:
             self.regressor = nn.Sequential(
                 nn.Linear(combined_dim, 256),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(256, 128),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(128, 1))

    def forward(self, sequence_embs, deltaGH):
        B, L = sequence_embs.shape
        device = sequence_embs.device

        # Positional encoding
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B,-1) # position indices

        # Embed sequences for encoder-decoder architecture
        seq_embed = self.token_embed(sequence_embs) + self.pos_embed(pos)
        on_target = seq_embed[:, :23, :]
        off_target = seq_embed[:, 24:, :]

        # Transformer expects (batch, seq, dim)
        out = self.transformer(on_target, off_target)  # shape [B, L, d_model]

        # Mean pooling sequence length
        pooled = out.mean(dim=1)
        
        # delta G embedding
        dg_emb = self.dg_embed(deltaGH)

        # Concatenate all features
        combined = torch.cat([pooled, dg_emb], dim=1)

        # Predict activity
        activity_pred = self.regressor(combined)

        return activity_pred.squeeze(-1)

def adjust_lr(optimizer, epoch):
    if epoch < 3:          # epochs 1,2,3 → 1e-3
        lr = 1e-3
    elif epoch < 10:        # epochs 4-10 → 1e-4
        lr = 1e-4
    else:                  # epochs  → 1e-5
        lr = 1e-5
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: makes dataloader workers deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)

def cross_val_layers(train_seq_embs, train_deltaGH_norm, train_activity,
                     num_layers=[1,2,3,4], k=5, num_epochs=5, batch_size=32):

    results={}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for num in num_layers:
        fold_spearman = []
        print(f'Testing num_layers={num}')

        for fold, (train_idx, cval_idx) in enumerate(kf.split(train_seq_embs)):
            print(f'Fold {fold+1}/{k}')

            X_train_seq = train_seq_embs[train_idx]
            X_train_delta = train_deltaGH_norm[train_idx]
            y_train = train_activity[train_idx]

            X_cval_seq = train_seq_embs[cval_idx]
            X_cval_delta = train_deltaGH_norm[cval_idx]
            y_cval = train_activity[cval_idx]

            train_dataset = TensorDataset(X_train_seq, X_train_delta, y_train)
            cval_dataset = TensorDataset(X_cval_seq, X_cval_delta, y_cval)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            cval_loader = DataLoader(cval_dataset, batch_size=batch_size, shuffle=False)

            model = CrossSeqTransformer(num_encoder_layers=num,
                                        num_decoder_layers=num)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(num_epochs):
                model.train()
                for seq_embs, deltaGH, activity in train_loader:
                    optimizer.zero_grad()
                    preds = model(seq_embs, deltaGH)
                    loss = criterion(preds, activity.squeeze(-1))
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            cval_preds = []
            cval_targets = []
            with torch.no_grad():
                for seq_embs, deltaGH, activity in cval_loader:
                    preds = model(seq_embs, deltaGH)
                    cval_preds.append(preds)
                    cval_targets.append(activity)

            cval_preds = torch.cat(cval_preds).numpy().squeeze()
            cval_targets = torch.cat(cval_targets).numpy().squeeze()

            rho, _ = spearmanr(cval_targets, cval_preds)
            fold_spearman.append(rho)
            print(f'Fold Spearman: {rho:.4f}')
        
        avg_rho = np.mean(fold_spearman)
        results[num] = avg_rho
        print(f'Average Spearman for num_layers={num}: {avg_rho:.4f}')

    return results

## -------- Training Loop --------- ##

# for dropout in dropout (add different dropout values to CrossSeqTransformer)
set_seed(42)
model = CrossSeqTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## NOTE: 10 epochs and batches for now
num_epochs = 10

train_seq_embs = train_data['sequence_embs']
train_delta = train_data['deltaGH'].unsqueeze(-1) if train_data['deltaGH'].dim() == 1 else train_data['deltaGH']
train_activity = train_data['activity'].unsqueeze(-1) if train_data['activity'].dim() == 1 else train_data['activity']
train_activity = np.log10(train_activity)

#train_dataset = TensorDataset(train_seq_embs, train_delta, train_activity)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Normalizing
train_deltaGH_min = train_data['deltaGH'].min()
train_deltaGH_max = train_data['deltaGH'].max()
train_deltaGH_norm = (train_delta - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min)   ## MAKE SURE NORMALIZATION IS CORRECT

val_seq_embs = val_data['sequence_embs']
val_deltaGH = val_data['deltaGH'].unsqueeze(-1) if val_data['deltaGH'].dim() == 1 else val_data['deltaGH']
val_deltaGH_norm = (val_deltaGH - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min) # Normalizing
val_activity = val_data['activity'].unsqueeze(-1) if val_data['activity'].dim() == 1 else val_data['activity']
val_activity = np.log10(val_activity)


val_dataset = TensorDataset(val_seq_embs, val_deltaGH_norm, val_activity)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_seq_embs = test_data['sequence_embs']
test_deltaGH = test_data['deltaGH'].unsqueeze(-1) if test_data['deltaGH'].dim() == 1 else test_data['deltaGH']
test_deltaGH_norm = (test_deltaGH - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min) # Normalizing
test_activity = test_data['activity'].unsqueeze(-1) if test_data['activity'].dim() == 1 else test_data['activity']
test_activity = np.log10(test_activity)

test_dataset = TensorDataset(test_seq_embs, test_deltaGH_norm, test_activity)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_dataset = TensorDataset(train_seq_embs, train_deltaGH_norm, train_activity)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# --- Cross-Validation --- #

num_layers_cv = [2, 3, 4, 5, 6]

cv_results = cross_val_layers(train_seq_embs, train_deltaGH_norm, train_activity,
                              num_layers = num_layers_cv, k=3, num_epochs=3)

print(cv_results)
