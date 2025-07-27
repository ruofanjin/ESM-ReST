from pydantic import BaseModel
from typing import List

class ModelConfig(BaseModel):
    input_dim: int
    hidden_dim: int
    dropout_rate: float
    num_heads: int
    use_cnn: bool
    use_attention: bool
    use_cross_attention: bool
    use_mutation_context: bool
    debug_attention: bool

class TrainingConfig(BaseModel):
    batch_size: int
    epochs_per_iteration: int
    lr: float
    criterion: str
    use_huber_loss: bool
    gradient_clip_threshold: float
    confidence_threshold: float
    early_stopping_patience: int

class ReSTConfig(BaseModel):
    num_iterations: int
    top_k_ratio: float
    keep_low_k: int
    blend_ratio: float
    stratify_sampling: bool
    lr_decay_factor: float
    lr_lower_bound: float
    freeze_backbone: bool

class GlobalConfig(BaseModel):
    data_path: str
    wt_seq: str
    device: str
    num_workers: int
    output_dir: str
    model: ModelConfig
    training: TrainingConfig
    rest: ReSTConfig