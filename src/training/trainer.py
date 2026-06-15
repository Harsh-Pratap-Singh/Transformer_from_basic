import torch
import torch.nn as nn
from tqdm import tqdm
import os
from .optimizer import create_optimizer_and_scheduler
from .metrics import compute_accuracy

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device,
                 src_tokenizer, tgt_tokenizer):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id, reduction='none')
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(model, config)
        self.best_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (src_ids, tgt_ids) in enumerate(pbar):
            src_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)

            # Teacher forcing: decoder input = tgt_ids without last token
            # target labels = tgt_ids without first token
            dec_input = tgt_ids[:, :-1]
            labels = tgt_ids[:, 1:]

            logits = self.model(src_ids, dec_input,
                                src_add_start=False, src_add_end=False,
                                tgt_add_start=False, tgt_add_end=False)

            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
            mask = labels.reshape(-1) != self.tgt_tokenizer.pad_id
            loss = loss.sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

            if (batch_idx + 1) % self.config.training.log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})

        return total_loss / total_tokens

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for src_ids, tgt_ids in tqdm(self.val_loader, desc="Validation"):
                src_ids = src_ids.to(self.device)
                tgt_ids = tgt_ids.to(self.device)

                dec_input = tgt_ids[:, :-1]
                labels = tgt_ids[:, 1:]

                logits = self.model(src_ids, dec_input,
                                    src_add_start=False, src_add_end=False,
                                    tgt_add_start=False, tgt_add_end=False)

                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.reshape(-1))
                mask = labels.reshape(-1) != self.tgt_tokenizer.pad_id
                loss = loss.sum() / mask.sum()

                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

        return total_loss / total_tokens

    def train(self):
        for epoch in range(self.config.training.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < self.best_loss and self.config.training.save_best:
                self.best_loss = val_loss
                os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(self.config.training.checkpoint_dir, 'best_model.pt'))