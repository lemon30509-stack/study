from dataclasses import dataclass


@dataclass
class GANConfig:
    z_dim: int = 100
    img_size: int = 224
    img_channels: int = 3
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_gp: float = 10.0
    n_critic: int = 5
    epochs: int = 200
    batch_size: int = 32


@dataclass
class CLSConfig:
    img_size: int = 224
    batch_size: int = 32
    max_epochs: int = 50
    patience: int = 10
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
