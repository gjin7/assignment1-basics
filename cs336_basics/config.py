from dataclasses import dataclass, field

@dataclass
class DataConfig:
    train_data_path: str = "data/tinystories_train.uint16.bin"
    val_data_path: str = "data/tinystories_val.uint16.bin"
    dtype: str = "uint16"

@dataclass 
class RunConfig:
    run_dir: str = "runs"
    run_name_prefix: str = "ts"
    resume_from: str | None = None
    
@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0

@dataclass
class TrainingConfig:
    max_steps: int = 10_000
    batch_size: int = 32

    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20
    
    ckpt_interval: int = 2000 
    device: str = "auto"
    seed: int = 1337

@dataclass 
class OptimizerConfig:
    lr_max: float = 3e-4
    lr_min: float = 1e-4
    warmup_steps: int = 2000 
    cosine_cycle_steps: int | None = None
    
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    max_grad_norm: float | None = 1.0 

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


def get_default_config():
    cfg = Config()
    return cfg


def get_mini_config():
    cfg = get_default_config()

    cfg.train.max_steps = 5000
    cfg.train.batch_size = 8
    cfg.train.log_interval = 10
    cfg.train.eval_interval = 100
    cfg.train.eval_batches = 2
    cfg.train.ckpt_interval = 1000

    cfg.model.context_length = 128
    cfg.model.num_layers = 2
    cfg.model.d_model = 128
    cfg.model.num_heads = 4
    cfg.model.d_ff = 384

    cfg.optimizer.warmup_steps = 30
    cfg.optimizer.cosine_cycle_steps = cfg.train.max_steps

    cfg.run.run_name_prefix = "ts_mini"
    return cfg
