import yaml
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    d_model: int = 512
    ffn_hidden: int = 2048
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    max_seq_len: int = 300

@dataclass
class DataConfig:
    train_ratio: float = 0.95
    batch_size: int = 30
    num_workers: int = 4
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None

@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 100
    save_best: bool = True

@dataclass
class InferenceConfig:
    beam_size: int = 4
    max_gen_len: int = 50

@dataclass
class VocabConfig:
    hindi_chars: List[str] = field(default_factory=list)
    english_chars: List[str] = field(default_factory=list)

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        inference=InferenceConfig(**raw["inference"]),
        vocab=VocabConfig(**raw["vocab"])
    )