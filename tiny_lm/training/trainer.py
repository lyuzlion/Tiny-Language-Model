"""
Training utilities for language models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable
import os
import json
import math


class Trainer:
    """
    Simple trainer for language models built from scratch.
    
    Handles training loop, evaluation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm: float = 1.0,
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000,
        output_dir: str = "./outputs",
    ):
        """
        Args:
            model: The language model to train
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
            optimizer: Optimizer (defaults to AdamW if not provided)
            lr_scheduler: Learning rate scheduler
            device: Device to train on
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: Steps between logging
            eval_interval: Steps between evaluations
            save_interval: Steps between checkpoints
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
        else:
            self.optimizer = optimizer
        
        self.lr_scheduler = lr_scheduler
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
    
    def train_step(self, batch: dict) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate scheduler step
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return loss.item()
    
    def evaluate(self) -> float:
        """
        Evaluate the model on the evaluation set.
        
        Returns:
            Average evaluation loss
        """
        if self.eval_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, filepath: str):
        """
        Save a checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load a checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        
        print(f"Checkpoint loaded from {filepath}")
    
    def train(
        self,
        num_epochs: int,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Optional path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Total training steps per epoch: {len(self.train_dataloader)}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step {self.global_step}: loss = {loss:.4f}, avg_loss = {avg_loss:.4f}, lr = {lr:.6f}")
                
                # Evaluation
                if self.eval_dataloader is not None and self.global_step % self.eval_interval == 0:
                    eval_loss = self.evaluate()
                    print(f"Evaluation at step {self.global_step}: eval_loss = {eval_loss:.4f}")
                    
                    # Save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        best_model_path = os.path.join(self.output_dir, "best_model.pt")
                        self.save_checkpoint(best_model_path)
                        print(f"New best model saved with eval_loss = {eval_loss:.4f}")
                
                # Checkpointing
                if self.global_step % self.save_interval == 0:
                    checkpoint_path = os.path.join(
                        self.output_dir,
                        f"checkpoint_step_{self.global_step}.pt"
                    )
                    self.save_checkpoint(checkpoint_path)
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} completed: avg_loss = {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(
                self.output_dir,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            self.save_checkpoint(epoch_checkpoint_path)
        
        print("\nTraining completed!")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as a ratio of initial lr
    
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
