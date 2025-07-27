import argparse
import yaml
from pathlib import Path
import pandas as pd
import torch
import esm
import numpy as np

# Import from our created modules
from .config import GlobalConfig
from .utils import preprocess_data, split_data
from .trainer import ReSTTrainer
from .dataset import ProteinDatasetESM2, collate_fn_esm2
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="ESM-ReST Protein Fitness Prediction")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the YAML config file")
    # Allow command line override of key parameters
    parser.add_argument('--lr', type=float, help="Override learning rate")
    parser.add_argument('--batch_size', type=int, help="Override batch size")
    args = parser.parse_args()

    # --- 1. Load configuration ---
    with open(args.config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.lr: config_dict['training']['lr'] = args.lr
    if args.batch_size: config_dict['training']['batch_size'] = args.batch_size
    
    config = GlobalConfig(**config_dict)
    print("--- Configuration loaded successfully ---")
    print(config.model_dump_json(indent=2))
    
    # --- 2. Prepare data ---
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    df = pd.read_csv(config.data_path)
    scaler_path = Path(config.output_dir) / "score_scaler.json"
    processed_df = preprocess_data(df, config.wt_seq, str(scaler_path))
    
    train_df, val_df = split_data(processed_df)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")
    
    # --- 3. Initialize model and trainer ---
    print("Loading ESM-2 model...")
    esm_model, esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    esm_model.eval()

    trainer = ReSTTrainer(config, esm_model, esm_alphabet)
    print(f"Experiment results will be saved in: {trainer.exp_dir}")

    # --- 4. Run training ---
    # ReSTTrainer will handle data loader creation internally
    trainer.run_rest(train_df, val_df)

    # --- 5. Final evaluation (optional) ---
    print("Training completed, loading best model for final evaluation...")
    # trainer.load_best_model() # You can implement this method in Trainer
    # trainer.evaluate(val_loader, prefix="final_test")

    print("✅ Full workflow completed.")

if __name__ == '__main__':
    main()