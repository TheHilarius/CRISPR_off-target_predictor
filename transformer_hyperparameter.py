#!/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import random
import os

train_data =torch.load("data/train_embeddings.pt")
val_data = torch.load("data/val_embeddings.pt")
test_data = torch.load("data/test_embeddings.pt")
       
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

def adjust_lr(optimizer, epoch, num_epochs):
    """
    Adaptive learning rate schedule that scales with total epochs.
    - First 30% of training (or first 3 epochs, whichever is longer): initial lr (1e-3)
    - Next 60% of training (or up to epoch 10, whichever is longer): lr / 10 (1e-4)
    - Final 10% of training: lr / 100 (1e-5)
    """
    threshold_1 = max(3, int(0.3 * num_epochs))
    threshold_2 = max(10, int(0.9 * num_epochs))
    
    if epoch < threshold_1:
        lr = 1e-3
    elif epoch < threshold_2:
        lr = 1e-4
    else:
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

## -------- HYPERPARAMETER TESTING --------- ##

# Define hyperparameter grid
hyperparameter_grid = {
    'num_epochs': [100], #10
    'dropout': [0.01, 0.05, 0.1, 0.2, 0.3], # Add 0.00 
    'batch_size': [16, 32]
}

# Store results
results = []
best_spearman = -np.inf
best_config = None

# Prepare data once
train_seq_embs = train_data['sequence_embs']
train_delta = train_data['deltaGH'].unsqueeze(-1) if train_data['deltaGH'].dim() == 1 else train_data['deltaGH']
train_activity = train_data['activity'].unsqueeze(-1) if train_data['activity'].dim() == 1 else train_data['activity']
train_activity = torch.log10(train_activity)

# Normalizing
train_deltaGH_min = train_data['deltaGH'].min()
train_deltaGH_max = train_data['deltaGH'].max()
train_deltaGH_norm = (train_delta - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min)

val_seq_embs = val_data['sequence_embs']
val_deltaGH = val_data['deltaGH'].unsqueeze(-1) if val_data['deltaGH'].dim() == 1 else val_data['deltaGH']
val_deltaGH_norm = (val_deltaGH - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min)
val_activity = val_data['activity'].unsqueeze(-1) if val_data['activity'].dim() == 1 else val_data['activity']
val_activity = torch.log10(val_activity)

test_seq_embs = test_data['sequence_embs']
test_deltaGH = test_data['deltaGH'].unsqueeze(-1) if test_data['deltaGH'].dim() == 1 else test_data['deltaGH']
test_deltaGH_norm = (test_deltaGH - train_deltaGH_min) / (train_deltaGH_max - train_deltaGH_min)
test_activity = test_data['activity'].unsqueeze(-1) if test_data['activity'].dim() == 1 else test_data['activity']
test_activity = torch.log10(test_activity)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter search loop
for num_epochs in hyperparameter_grid['num_epochs']:
    for dropout in hyperparameter_grid['dropout']:
        for batch_size in hyperparameter_grid['batch_size']:
            print(f"\n{'='*80}")
            print(f"Testing: epochs={num_epochs}, dropout={dropout}, batch_size={batch_size}")
            print(f"{'='*80}\n")
            
            # Reset seed for reproducibility
            set_seed(42)
            
            # Create model with current dropout
            model = CrossSeqTransformer(dropout=dropout)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model = model.to(device)
            
            # Create data loaders with current batch_size
            train_dataset = TensorDataset(train_seq_embs, train_deltaGH_norm, train_activity)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            val_dataset = TensorDataset(val_seq_embs, val_deltaGH_norm, val_activity)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            test_dataset = TensorDataset(test_seq_embs, test_deltaGH_norm, test_activity)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            ## -------- Training Loop --------- ##
            all_preds = []
            all_targets = []
            
            for epoch in range(num_epochs):
                
                current_lr = adjust_lr(optimizer, epoch, num_epochs)
                
                epoch_loss = 0
                num_batches = 0
                
                # containers for THIS epoch only
                epoch_preds = []
                epoch_targets = []
                epoch_spearman = []
                
                for sequence_embs, deltaGH, activity in train_loader:
                    sequence_embs = sequence_embs.to(device)
                    deltaGH = deltaGH.to(device)
                    activity = activity.to(device)
                    
                    predictions = model(sequence_embs, deltaGH)
                    loss = criterion(predictions, activity.squeeze(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # ---- save predictions for this epoch ----
                    epoch_preds.append(predictions.detach().cpu())
                    epoch_targets.append(activity.detach().cpu())

                # ---- combine epoch-level tensors ----
                epoch_preds = torch.cat(epoch_preds).squeeze().numpy()
                epoch_targets = torch.cat(epoch_targets).squeeze().numpy()

                # ---- compute Spearman correlation ----
                rho, p_value = spearmanr(epoch_targets, epoch_preds)
                epoch_spearman.append(rho)

                # ---- Compute training AUC ----
                threshold = np.median(epoch_targets)
                y_true = (epoch_targets >= threshold).astype(int)
                y_score = epoch_preds
                train_auc = roc_auc_score(y_true, y_score)

                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"loss={epoch_loss/num_batches:.4f}, "
                      f"Spearman rho={rho:.4f}, "
                      f"AUC={train_auc:.4f}")

            # Evaluate on validation set (AFTER all epochs complete)
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0
            val_num_batches = 0
            
            with torch.no_grad():
                for sequence_embs, deltaGH, activity in val_loader:
                    sequence_embs = sequence_embs.to(device)
                    deltaGH = deltaGH.to(device)
                    activity = activity.to(device)
                    
                    predictions = model(sequence_embs, deltaGH)
                    loss = criterion(predictions, activity.squeeze(-1))
                    
                    val_loss += loss.item()
                    val_num_batches += 1
                    
                    val_preds.append(predictions.cpu())
                    val_targets.append(activity.cpu())
            
            val_preds = torch.cat(val_preds).squeeze().numpy()
            val_targets = torch.cat(val_targets).squeeze().numpy()
            val_spearman, _ = spearmanr(val_targets, val_preds)
            val_loss_avg = val_loss / val_num_batches
            
            # ---- Compute validation AUC ----
            threshold_val = np.median(val_targets)
            y_true_val = (val_targets >= threshold_val).astype(int)
            y_score_val = val_preds
            val_auc = roc_auc_score(y_true_val, y_score_val)
            
            print(f"\nValidation Loss: {val_loss_avg:.4f}, Validation Spearman: {val_spearman:.4f}, Validation AUC: {val_auc:.4f}")
            
            # Store results
            result = {
                'num_epochs': num_epochs,
                'dropout': dropout,
                'batch_size': batch_size,
                'final_train_loss': epoch_loss/num_batches,
                'final_train_spearman': rho,
                'final_train_auc': train_auc,
                'val_loss': val_loss_avg,
                'val_spearman': val_spearman,
                'val_auc': val_auc
            }
            results.append(result)
            
            # Track best configuration
            if val_spearman > best_spearman:
                best_spearman = val_spearman
                best_config = result.copy()
                print(f"\n*** NEW BEST CONFIG! Val Spearman: {val_spearman:.4f} ***")

print("\n" + "="*80)
print("HYPERPARAMETER SEARCH COMPLETE")
print("="*80)

# Display results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_spearman', ascending=False)

print("\nAll Results (sorted by validation Spearman):")
print(results_df.to_string(index=False))

print(f"\n{'='*80}")
print("BEST CONFIGURATION:")
print(f"{'='*80}")
for key, value in best_config.items():
    print(f"{key}: {value}")

# Save results to CSV
results_df.to_csv('hyperparameter_results1.csv', index=False)
print("\nResults saved to 'hyperparameter_results.csv'")

print("Training complete")