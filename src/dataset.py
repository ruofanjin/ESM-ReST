import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from typing import List, Optional
import gc

class ProteinDatasetESM2(Dataset):
    """Protein sequence dataset using ESM-2 embeddings with caching."""
    def __init__(self, df: pd.DataFrame, esm_model, esm_alphabet, device='cpu', max_cache_size=1000):
        self.df = df
        self.esm_model = esm_model
        self.esm_alphabet = esm_alphabet
        self.device = device
        self.max_len = 1022
        self.max_cache_size = max_cache_size
        self.embedding_cache = {}

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            wt_seq = ''.join(row['wt_seq']).replace('*', '')
            mut_seq = ''.join(row['mut_seq']).replace('*', '')
            
            wt_seq_trunc = wt_seq[:self.max_len]
            mut_seq_trunc = mut_seq[:self.max_len]
            
            wt_key = f"wt_{wt_seq_trunc}"
            mut_key = f"mut_{mut_seq_trunc}_{idx}"
            
            if wt_key not in self.embedding_cache:
                self.embedding_cache[wt_key] = self._get_embedding(wt_seq_trunc)
            
            if mut_key not in self.embedding_cache:
                self.embedding_cache[mut_key] = self._get_embedding(mut_seq_trunc)
            
            wt_embedding = self.embedding_cache[wt_key].to(self.device)
            mut_embedding = self.embedding_cache[mut_key].to(self.device)
            
            pos_str = row['pos']
            positions = [int(p) - 1 for p in pos_str.split(',')] if pos_str else []
            positions = [p for p in positions if 0 <= p < len(wt_seq_trunc)]
            
            score = torch.unsqueeze(torch.FloatTensor([row['score']]), 0)

            if len(self.embedding_cache) > self.max_cache_size:
                self._clear_cache()

            return wt_embedding, mut_embedding, positions, len(wt_seq_trunc), score
        except Exception as e:
            print(f"Error in __getitem__ for idx {idx}: {e}")
            return self._get_default_sample()

    def _get_embedding(self, sequence: str) -> torch.Tensor:
        batch_converter = self.esm_alphabet.get_batch_converter()
        data = [("seq", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[12], return_contacts=False)
            return results["representations"][12][0, 1:-1, :].cpu()

    def _clear_cache(self):
        # Clear half of the cache to avoid frequent clearing
        keys_to_del = list(self.embedding_cache.keys())[:len(self.embedding_cache)//2]
        for key in keys_to_del:
            del self.embedding_cache[key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_default_sample(self):
        default_embedding = torch.zeros((100, 480), device=self.device)
        return default_embedding, default_embedding, [], 100, torch.unsqueeze(torch.FloatTensor([0.0]), 0)

    def __len__(self):
        return len(self.df)

def collate_fn_esm2(batch):
    try:
        wt_embeddings, mut_embeddings, positions_list, lengths, labels = zip(*batch)

        wt_padded = pad_sequence([seq.clone().detach() for seq in wt_embeddings], batch_first=True, padding_value=0)
        mut_padded = pad_sequence([seq.clone().detach() for seq in mut_embeddings], batch_first=True, padding_value=0)

        max_positions = max(len(pos) for pos in positions_list) if any(positions_list) else 1
        padded_positions, position_masks = [], []
        for pos_list in positions_list:
            pad_len = max_positions - len(pos_list)
            padded_positions.append(pos_list + [0] * pad_len)
            position_masks.append([1] * len(pos_list) + [0] * pad_len)
        
        pos = torch.tensor(padded_positions, dtype=torch.long)
        pos_mask = torch.tensor(position_masks, dtype=torch.bool)
        
        length_tensor = torch.tensor(lengths, dtype=torch.long)
        labels_tensor = torch.stack(labels)

        return wt_padded, mut_padded, pos, pos_mask, length_tensor, labels_tensor
    except Exception as e:
        print(f"Collate error: {e}")
        return None