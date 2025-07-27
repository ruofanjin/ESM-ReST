import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import numpy as np
import hashlib
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional

from .dataset import ProteinDatasetESM2, collate_fn_esm2
from .model import ESM2Effect
from .utils import (
    plot_reward_distribution, 
    plot_correlation, 
    plot_error_by_mutation_count
)

class FocalLoss(nn.Module):
    """Focal Loss for regression tasks."""
    def __init__(self, alpha=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        mse_loss = F.mse_loss(inputs, targets, reduction='none')
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * mse_loss
        return focal_loss.mean()

class ReSTTrainer:
    """
    Trainer that encapsulates ReST self-training loop and core training/evaluation logic.
    """
    def __init__(self, config, esm_model, esm_alphabet):
        self.config = config
        self.device = torch.device(config.device)
        self.esm_model = esm_model.to(self.device)
        self.esm_alphabet = esm_alphabet
        
        self.model = self._init_model()
        
        self.exp_dir = Path(config.output_dir) / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir, self.model_dir, self.results_dir = self._setup_directories()
            
        self.writer = SummaryWriter(self.log_dir)
        self.scaler = GradScaler()
        
        self.focal_loss = FocalLoss()
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.1)
        self.confidence_criterion = nn.BCEWithLogitsLoss()
        
        self.best_val_metric = -float('inf')
        self.best_model_path = None

    def _init_model(self):
        model = ESM2Effect(self.config.model)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        return model.to(self.device)

    def _setup_directories(self):
        dirs = [self.exp_dir, self.exp_dir / "logs", self.exp_dir / "models", self.exp_dir / "results"]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        # Save configuration to experiment directory
        with open(self.exp_dir / "config.yaml", 'w') as f:
            f.write(self.config.model_dump_json(indent=2))
        return dirs[1], dirs[2], dirs[3]

    def _setup_optimizer_and_scheduler(self, train_loader, lr):
        param_groups = [{'params': p, 'lr': lr} for p in self.model.parameters()]
        self.optimizer = AdamW(param_groups, weight_decay=0.01)
        self.scheduler = OneCycleLR(
            self.optimizer, max_lr=lr, 
            total_steps=len(train_loader) * self.config.training.epochs_per_iteration,
            pct_start=0.1
        )

    def _process_batch(self, batch, training: bool) -> Dict:
        """Process a single batch, compute loss and predictions."""
        *model_inputs, labels = [b.to(self.device) for b in batch]
        labels = labels.float().squeeze(-1)
        
        with autocast(enabled=(self.config.device == 'cuda')):
            score_preds, conf_preds = self.model(*model_inputs)
            score_preds, conf_preds = score_preds.view(-1), conf_preds.view(-1)
            
            score_loss = self.focal_loss(score_preds, labels)
            
            ranking_loss = torch.tensor(0.0, device=self.device)
            if len(labels) > 1:
                idx_i, idx_j = torch.triu_indices(len(labels), len(labels), 1)
                target = torch.sign(labels[idx_i] - labels[idx_j])
                ranking_loss = self.ranking_criterion(score_preds[idx_i], score_preds[idx_j], target)
                
            errors = torch.abs(score_preds - labels)
            conf_targets = (errors < errors.mean() + 0.5 * errors.std()).float()
            conf_loss = self.confidence_criterion(conf_preds, conf_targets)
            
            loss = score_loss + 0.2 * ranking_loss + 0.15 * conf_loss

        if training and loss.isfinite():
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip_threshold)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler: self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() if loss.isfinite() else float('inf'),
            "score_preds": score_preds.detach().cpu().numpy(),
            "conf_preds": torch.sigmoid(conf_preds).detach().cpu().numpy(),
            "labels": labels.detach().cpu().numpy(),
            "num_mutations": batch[3].sum(dim=1).cpu().numpy()
        }

    def _run_epoch(self, data_loader, training: bool, current_epoch: int = 0):
        """Run a complete epoch (training or evaluation)."""
        self.model.train(training)
        epoch_loss = 0
        all_results = {"score_preds": [], "conf_preds": [], "labels": [], "num_mutations": []}
        
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in data_loader:
                if batch is None: continue
                batch_results = self._process_batch(batch, training=training)
                epoch_loss += batch_results["loss"]
                for key in all_results:
                    all_results[key].extend(batch_results[key])
        
        avg_loss = epoch_loss / len(data_loader) if data_loader else 0
        for key in all_results:
            all_results[key] = np.concatenate(all_results[key]) if all_results[key] else np.array([])
        
        return avg_loss, all_results

    def _evaluate_metrics(self, results: Dict, prefix: str) -> Dict:
        """Calculate all evaluation metrics based on results from one epoch."""
        df = pd.DataFrame({
            'score': results['labels'],
            'prediction': results['score_preds'],
            'confidence': results['conf_preds'],
            'num_mutations': results['num_mutations']
        })
        if len(df) < 2: return {f"{prefix}_loss": float('inf')}

        metrics = {}
        metrics[f'{prefix}_pearson'], _ = pearsonr(df['score'], df['prediction'])
        metrics[f'{prefix}_spearman'], _ = spearmanr(df['score'], df['prediction'])

        for (low, high) in [(5, 8), (9, 12), (13, 20), (21, 100)]:
            subset = df[df['num_mutations'].between(low, high)]
            if len(subset) > 1:
                p, _ = pearsonr(subset['score'], subset['prediction'])
                metrics[f'{prefix}_{low}-{high}_pearson'] = p
        return metrics

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on the given dataframe."""
        dataset = ProteinDatasetESM2(dataframe, self.esm_model, self.esm_alphabet, self.device)
        data_loader = DataLoader(
            dataset, batch_size=self.config.training.batch_size, 
            num_workers=self.config.num_workers, collate_fn=collate_fn_esm2
        )
        _, results = self._run_epoch(data_loader, training=False)
        dataframe['prediction'] = results['score_preds']
        dataframe['confidence'] = results['conf_preds']
        return dataframe

    def run_rest(self, train_df, val_df):
        """Execute the complete ReST self-training workflow."""
        current_train_df = train_df.copy()
        global_patience_counter = 0

        for iteration in range(self.config.rest.num_iterations):
            print(f"\n{'='*20} ReST Iteration {iteration + 1}/{self.config.rest.num_iterations} {'='*20}")
            
            # --- ReST Step 1: Reward computation ---
            rewarded_df = self._compute_rewards(current_train_df)
            plot_reward_distribution(rewarded_df, iteration + 1, self.results_dir)

            # --- ReST Step 2: Sample filtering ---
            top_k = int(len(rewarded_df) * self.config.rest.top_k_ratio)
            filtered_df = self._filter_high_reward_samples(rewarded_df, top_k)
            
            # --- ReST Step 3: Update training set ---
            current_train_df = pd.concat([current_train_df, filtered_df]).drop_duplicates().sample(frac=1)
            print(f"New training set size: {len(current_train_df)}")

            # --- ReST Step 4: Internal training loop ---
            train_loader = DataLoader(
                ProteinDatasetESM2(current_train_df, self.esm_model, self.esm_alphabet, self.device),
                batch_size=self.config.training.batch_size, shuffle=True, 
                num_workers=self.config.num_workers, collate_fn=collate_fn_esm2
            )
            val_loader = DataLoader(
                ProteinDatasetESM2(val_df, self.esm_model, self.esm_alphabet, self.device),
                batch_size=self.config.training.batch_size,
                num_workers=self.config.num_workers, collate_fn=collate_fn_esm2
            )
            
            # Learning rate decay
            current_lr = max(self.config.training.lr * (self.config.rest.lr_decay_factor ** iteration), self.config.rest.lr_lower_bound)
            self._setup_optimizer_and_scheduler(train_loader, current_lr)
            print(f"Set learning rate for this iteration to: {current_lr:.6f}")

            iteration_best_metric = -float('inf')
            patience_counter = 0

            for epoch in range(self.config.training.epochs_per_iteration):
                train_loss, _ = self._run_epoch(train_loader, training=True, current_epoch=epoch)
                val_loss, val_results = self._run_epoch(val_loader, training=False)
                metrics = self._evaluate_metrics(val_results, "val")
                
                print(f"Iter {iteration+1} Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Pearson={metrics.get('val_pearson', 0):.4f}")
                
                # Log to TensorBoard
                self.writer.add_scalar(f"Iter_{iteration+1}/Train_Loss", train_loss, epoch)
                for name, value in metrics.items():
                    self.writer.add_scalar(f"Iter_{iteration+1}/{name.capitalize()}", value, epoch)

                # Save model and early stopping
                current_metric = metrics.get('val_pearson', -float('inf'))
                if current_metric > iteration_best_metric:
                    iteration_best_metric = current_metric
                    patience_counter = 0
                    if current_metric > self.best_val_metric:
                        self.best_val_metric = current_metric
                        self.best_model_path = self.model_dir / f"best_model_overall_iter{iteration+1}.pt"
                        torch.save(self.model.state_dict(), self.best_model_path)
                        print(f"🚀 New best overall model saved with Pearson: {self.best_val_metric:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= self.config.training.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
            
            # Check if global performance has improved
            if iteration > 0 and iteration_best_metric < self.best_val_metric:
                global_patience_counter += 1
            else:
                global_patience_counter = 0

            if global_patience_counter >= 2:
                print("Global performance has not improved for 2 iterations. Stopping ReST.")
                break
        
        # --- Final evaluation ---
        print("\nReST training finished. Loading best model for final evaluation.")
        if self.best_model_path and self.best_model_path.exists():
            self.model.load_state_dict(torch.load(self.best_model_path))
            val_loader = DataLoader(
                ProteinDatasetESM2(val_df, self.esm_model, self.esm_alphabet, self.device),
                batch_size=self.config.training.batch_size, num_workers=self.config.num_workers, collate_fn=collate_fn_esm2
            )
            final_predictions = self.predict(val_df)
            plot_correlation(final_predictions, 'score', 'prediction', 'Final Model Performance', self.results_dir / 'final_correlation.png')
            plot_error_by_mutation_count(final_predictions, self.results_dir / 'final_error_by_mutation.png')
            print(f"Final plots saved to {self.results_dir}")
        else:
            print("No best model was saved. Skipping final evaluation.")

    def _compute_rewards(self, df):
        print("Computing rewards for ReST...")
        pred_df = self.predict(df.copy())
        
        pearson, _ = pearsonr(pred_df['score'], pred_df['prediction'])
        spearman, _ = spearmanr(pred_df['score'], pred_df['prediction'])
        combined_corr = 0.5 * pearson + 0.5 * spearman
        
        # Blend true labels and predictions as reward
        pred_df['reward'] = (
            (1 - self.config.rest.blend_ratio) * pred_df['prediction'] +
            self.config.rest.blend_ratio * pred_df['score']
        )
        pred_df['reward'] += combined_corr * 0.05 # Slightly adjust reward using correlation
        return pred_df

    def _filter_high_reward_samples(self, df, top_k):
        print(f"Filtering top {top_k} samples based on reward...")
        if self.config.rest.stratify_sampling:
            df['reward_quantile'] = pd.qcut(df['reward'], q=10, labels=False, duplicates='drop')
            # Stratified sampling by quantile and mutation count to ensure diversity
            return df.groupby('reward_quantile').apply(lambda x: x.sample(frac=0.1, random_state=42)).nlargest(top_k, 'reward')
        else:
            return df.nlargest(top_k, 'reward')