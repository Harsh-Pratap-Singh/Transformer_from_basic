import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def create_optimizer_and_scheduler(model, config):
    optimizer = AdamW(model.parameters(),
                      lr=config.training.learning_rate,
                      weight_decay=config.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    return optimizer, scheduler