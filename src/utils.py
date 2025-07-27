import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

def find_mutation_positions(wt_seq: str, mut_seq: str) -> str:
    """Compare two sequences and return mutation positions as string (1-based index)."""
    return ','.join([str(i + 1) for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)) if wt != mut])

def preprocess_data(df: pd.DataFrame, wt_seq: str, scaler_path: Path) -> pd.DataFrame:
    """
    Complete preprocessing of the raw DataFrame.
    Includes: finding mutation sites, removing wild type, log10 transformation, Z-score normalization, and adding auxiliary columns.
    """
    print("Starting data preprocessing...")
    df['pos'] = df['AminoAcidSequence'].apply(lambda x: find_mutation_positions(wt_seq, x))
    df = df[df['pos'] != ''].reset_index(drop=True)
    
    # Label processing: log10 + z-score
    df['score_raw'] = np.log10(df['Count'].clip(lower=1e-8)) # Use clip to avoid log(0)
    
    scaler = StandardScaler()
    df['score'] = scaler.fit_transform(df['score_raw'].values.reshape(-1, 1)).flatten()
    
    # Save scaler parameters for later use
    scaler_params = {'mean': float(scaler.mean_[0]), 'std': float(scaler.scale_[0])}
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)
    print(f"Label scaling parameters saved to {scaler_path}")
    
    df['wt_seq'] = wt_seq
    df['mut_seq'] = df['AminoAcidSequence']
    df['num_mutations'] = df['pos'].apply(lambda x: len(x.split(',')) if x else 0)
    
    return df

def split_data(df: pd.DataFrame):
    """Split data into training and validation sets with stratification."""
    # Use pd.qcut for stratification, adding robustness
    try:
        strata = pd.qcut(df['score'], q=10, labels=False, duplicates='drop')
    except ValueError:
        # If stratification fails (e.g., too few data points), don't stratify
        strata = None

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=strata
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --- Plotting Functions ---

def plot_correlation(df, x_col, y_col, title, save_path, confidence_threshold=0.3):
    """Plot correlation scatter plot between predicted and true values."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ev = df.dropna(subset=[x_col, y_col, 'confidence'])
    if ev.empty:
        print("Warning: No data to plot for correlation.")
        return

    pearson_corr, _ = pearsonr(ev[x_col], ev[y_col])
    spearman_corr, _ = spearmanr(ev[x_col], ev[y_col])

    ax.scatter(ev[x_col], ev[y_col], alpha=0.3, label=f"All ({len(ev)})")
    
    high_conf_ev = ev[ev['confidence'] >= confidence_threshold]
    if not high_conf_ev.empty:
        ax.scatter(high_conf_ev[x_col], high_conf_ev[y_col], alpha=0.5, color='red', label=f'High Conf (≥{confidence_threshold}, {len(high_conf_ev)})')

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y=x')
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    textstr = '\n'.join([
        f'Spearman: {spearman_corr:.3f}',
        f'Pearson: {pearson_corr:.3f}',
    ])
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("True Score (Normalized)", fontsize=12)
    ax.set_ylabel("Predicted Score (Normalized)", fontsize=12)
    ax.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_error_by_mutation_count(df: pd.DataFrame, save_path: str):
    """Plot box plot of prediction errors by number of mutations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    df['error'] = (df['score'] - df['prediction']).abs()
    
    data_to_plot = df.groupby('num_mutations')['error'].apply(list).sort_index()
    if data_to_plot.empty:
        print("Warning: No data to plot for error by mutation count.")
        return

    ax.boxplot(data_to_plot.values, labels=data_to_plot.index)
    
    ax.set_title('Prediction Error by Number of Mutations', fontsize=16)
    ax.set_xlabel('Number of Mutations', fontsize=12)
    ax.set_ylabel('Absolute Prediction Error', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_reward_distribution(df: pd.DataFrame, iteration: int, save_path: str):
    """Plot histogram of reward distribution."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df['reward'], bins=50, alpha=0.8, color='skyblue', edgecolor='black')
    
    ax.set_title(f'Reward Distribution - ReST Iteration {iteration}', fontsize=16)
    ax.set_xlabel('Reward Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)