#/usr/bin/env python3

# https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.transformer.Transformer.html

import torch
# install torch
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
from torch import nn

class CrossSeqTransformer(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 max_len=32, dropout=0.1, dg_embedding_dim=16):
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
            nn.Linear(d_model + d_model // 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
            nn.Sigmoid() # activity between 0 and 1
        )

    def forward(self, on_seq, off_seq, delta_g):
        B, L = on_seq.shape
        device = on_seq.device

        # Positional encoding
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B,-1) # position indices

        # Embed sequences for encoder-decoder architecture
        src = self.token_embed(on_seq) + self.pos_embed(pos)
        tgt = self.token_embed(off_seq) + self.pos_embed(pos)

        # Transformer expects (batch, seq, dim)
        out = self.transformer(src, tgt)  # shape [B, L, d_model]
        
        # Mean pooling sequence length
        pooled = out.mean(dim=1)
        
        # delta G embedding
        dg_emb = self.dg_embed(delta_g)

        # Concatenate all features
        combined = torch.cat([pooled, dg_emb], dim=1)

        # Predict activity
        activity_pred = self.regressor(combined)

        return activity_pred.squeeze(-1)
    
# ---------- Training example------------ #

model = CrossSeqTransformer()
batch_size = 32
seq_len = 24

# Dummy input
on_seq = torch.randint(0, 5, (batch_size, seq_len))
off_seq = torch.randint(0, 5, (batch_size, seq_len))
delta_g = torch.rand(batch_size, 1)
activity_scores = torch.rand(batch_size)

# Forward pass
out = model(on_seq, off_seq, delta_g)
print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")

## -------- Training Loop --------- ##

model = CrossSeqTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

## NOTE: 10 epochs and batches for now
num_batches = 10
for epoch in range(10):
    epoch_loss = 0.0
    for _ in range(num_batches):
        on_seq = torch.randint(0, 5, (batch_size, seq_len))
        off_seq = torch.randint(0, 5, (batch_size, seq_len))
        delta_g = torch.rand(batch_size, 1)
        activity_scores = torch.rand(batch_size)

        # Forward pass
        out = model(on_seq, off_seq, delta_g)
        loss = criterion(out, activity_scores)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: avg loss = {epoch_loss / num_batches:.4f}")

print("Training complete")