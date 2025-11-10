# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 12:33:03 2025

@author: zbs768
"""

import pandas as pd
import torch


train = pd.read_csv("data/train_set.csv")
test = pd.read_csv("data/test_set.csv")
val = pd.read_csv("data/validation_set.csv")
ext_test = pd.read_csv("data/ext_test_set.csv")


BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "[SEP]": 5, "[PAD]": 6}

def tokenize_seq(seq):
    """Convert a 23bp sequence string into a list of token IDs."""
    return [BASE2IDX.get(base, 4) for base in seq]


def preprocess_dataframe(df, ontarget, offtarget, deltaGH, activity, max_len=47):
    """
    Converts a CRISPR DataFrame into tensor batches.
    Expects columns: Ontarget, Offtarget, DeltaGH, Activity
    """
    input_ids, deltas, labels = [], [], []

    for _, row in df.iterrows():
        # Combine on/off target sequences with a separator
        on = [BASE2IDX[b] for b in row[ontarget]]
        sep = [BASE2IDX["[SEP]"]]
        off = [BASE2IDX[b] for b in row[offtarget]]
        
        token_ids = on + sep + off
        

        # Pad or trim to fixed length
        if len(token_ids) < max_len:
            token_ids += [BASE2IDX["[PAD]"]] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]

        input_ids.append(token_ids)
        deltas.append([row[deltaGH]])
        labels.append([row[activity]])

        #Convert to tensors  
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    deltas = torch.tensor(deltas, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return {"sequence_embs": input_ids, "deltaGH": deltas, "activity": labels}


train_emb = preprocess_dataframe(df = train, ontarget="target", 
                                 offtarget="offtarget_sequence",
                                 activity="CHANGEseq_reads",
                                 deltaGH="DeltaGH")
test_emb = preprocess_dataframe(df = test, ontarget="target", 
                                 offtarget="offtarget_sequence",
                                 activity="CHANGEseq_reads",
                                 deltaGH="DeltaGH")
val_emb = preprocess_dataframe(df = val, ontarget="target", 
                                 offtarget="offtarget_sequence",
                                 activity="CHANGEseq_reads",
                                 deltaGH="DeltaGH")

ext_test_emb = preprocess_dataframe(df = ext_test, ontarget="Target_Sequence", 
                                 offtarget="Offtarget_Sequence",
                                 activity="GUIDEseq_Reads",
                                 deltaGH="DeltaGH")

torch.save(train_emb, "data/train_embeddings.pt")
torch.save(val_emb, "data/val_embeddings.pt")
torch.save(test_emb, "data/test_embeddings.pt")
torch.save(ext_test_emb, "data/ext_test_embeddings.pt")