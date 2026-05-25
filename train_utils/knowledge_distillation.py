import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm

import transformers
import wandb
import collections.abc

from utils import data_utils, eval_utils, utils


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 1.0  # Temperature for softmax
    alpha: float = 0.7  # Weight for distillation loss (vs task loss)
    batch_size: int = 8
    learning_rate: float = 1e-6
    num_epochs: int = 3
    max_seq_length: int = 2048
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KnowledgeDistiller:
    """Knowledge Distillation trainer for language models"""

    def __init__(self, student, teacher, tokenizer, config: DistillationConfig, ptq_args):
        self.config = config
        self.device = torch.device(config.device)

        # Load teacher model
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Load student model
        self.student = student
        self.student.train()
        self.args = ptq_args

        # Load tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate
        )
        self.scheduler = None # will be defined later

    from typing import Tuple
    import torch
    import torch.nn.functional as F

    def distillation_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            targets: torch.Tensor,
            temperature: float = 4.0
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate distillation loss
        """
        # Force computations onto the student's device to avoid cross-device grad issues
        target_device = student_logits.device
        teacher_logits = teacher_logits.to(target_device)
        targets = targets.to(target_device)
        
        # 1. Filter out ignored tokens (-100) before KL Div
        valid_mask = targets != -100
        
        # SAFETY CHECK: If the entire batch is masked out, return zero losses
        if not valid_mask.any():
            zero_loss = torch.tensor(0.0, device=target_device, requires_grad=True)
            return zero_loss, zero_loss.detach(), zero_loss.detach()

        # Apply the mask
        student_logits = student_logits[valid_mask]
        teacher_logits = teacher_logits[valid_mask]
        targets = targets[valid_mask]

        # 2. Distillation loss (KL divergence with temperature)
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

        # Note: reduction="batchmean" mathematically acts as "mean" over the valid tokens 
        # here since we flattened and masked our tensors. This is the correct behavior.
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction="batchmean"
        ) * (temperature ** 2)

        # 3. Task loss (cross-entropy with ground truth)
        task_loss = F.cross_entropy(student_logits, targets)

        # Combined loss
        total_loss = self.args.kd_alpha * kl_loss + (1 - self.args.kd_alpha) * task_loss

        return total_loss, kl_loss, task_loss


    def perform_training_step(self, batch, use_distillation=True):
        """Perform a single training step"""
        
        # 1. Flexible batch unpacking
        if isinstance(batch, collections.abc.Mapping):
            input_ids = batch['input_ids']
            labels = batch.get('labels', None)
        else:
            # Fallback for standard tuple datasets
            input_ids = batch[0].squeeze(1) if batch[0].dim() > 2 else batch[0]
            labels = batch[1] if len(batch) > 1 else None 
        
        # Forward pass through teacher (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids.to(self.teacher.device),
                output_hidden_states=False,
            )
            teacher_logits = teacher_outputs.logits

        # Forward pass through student
        student_outputs = self.student(
            input_ids=input_ids.to(self.student.device),
            output_hidden_states=False,
        )
        student_logits = student_outputs.logits

        # 2. Contextual target selection & Shifting
        # Causal LM requires shifting logits to predict the *next* token
        if labels is not None:
            targets = labels[:, 1:].contiguous()
            
            if len(targets) == 0:
                print("Warning: Labels provided but empty after shifting. Falling back to input_ids for targets.")
                targets = input_ids[:, 1:].contiguous()
        else:
            # Note: If falling back to input_ids, make sure your input_ids don't contain 
            # padding tokens, or the model will learn to predict padding from padding!
            targets = input_ids[:, 1:].contiguous()

        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()

        # Reshape for loss computation
        batch_size, seq_len, vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, vocab_size)
        teacher_logits_flat = teacher_logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        total_training_tokens = (targets_flat != -100).sum().item()

        # Calculate losses
        loss, kl_loss, task_loss = self.distillation_loss(
            student_logits_flat,
            teacher_logits_flat,
            targets_flat,
            temperature=self.config.temperature
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        if use_distillation:
            loss.backward()
        else:
            task_loss.backward()
            
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        self.optimizer.step()            
            
        # Memory management: Delete large tensors, but DO NOT use empty_cache()
        del student_logits, teacher_logits, student_logits_flat, teacher_logits_flat, targets_flat
        return loss, kl_loss, task_loss, total_training_tokens

    def train_epoch(self, train_loader: DataLoader, use_distillation=True, use_alternating_LR_training=False) -> dict:
        """Train for one epoch"""
        self.student.train()
        print(f"\nTraining for one epoch with distillation: {use_distillation} and alternating LR training: {use_alternating_LR_training} ...")
        
        # enable only SVDLinear layers to be trained
        print("Freezing all student parameters except SVDLinear layers...")
        for name, param in self.student.named_parameters():
            param.requires_grad = False
            if "ALinear" in name or "BLinear" in name:
                param.requires_grad = True
        
        total_loss = 0
        total_kl_loss = 0 
        total_task_loss = 0
        batch_counter = 0
        total_train_tokens = 0
        
        num_batches = len(train_loader)
        eval_every = None
        evals_per_epoch = self.args.num_evals_per_epoch_zeroshot
        if evals_per_epoch is not None and evals_per_epoch > 0:
            eval_every = max(1, num_batches // evals_per_epoch)

        for batch_idx, batch in tqdm(enumerate(train_loader), f"Training with distillation: {use_distillation} ..."):
            
            if use_alternating_LR_training:
                self.enable_left_components()
                loss, kl_loss, task_loss, training_step_tokens = self.perform_training_step(batch, use_distillation)
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                total_train_tokens += training_step_tokens

                batch_counter += 1
                
                self.enable_right_components()
                loss, kl_loss, task_loss, training_step_tokens = self.perform_training_step(batch, use_distillation)
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                total_train_tokens += training_step_tokens
                batch_counter += 1
            
            else:
                # Perform training step
                loss, kl_loss, task_loss, training_step_tokens = self.perform_training_step(batch, use_distillation)
                # Track metrics
                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_task_loss += task_loss.item()
                total_train_tokens += training_step_tokens
                batch_counter += 1
            
            # Step it here to avoid stepping twice in alternating LR training
            if self.scheduler is not None:
                self.scheduler.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"KL Loss: {kl_loss.item():.4f} | "
                    f"Task Loss: {task_loss.item():.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.8f}"
                    f" | Tokens: {total_train_tokens}"
                )
                
            if eval_every is not None and (batch_idx + 1) % eval_every == 0:
                eval_utils.run_subset_benchmarks(self.student, self.tokenizer, self.device, limit=100)
                gc.collect()
                torch.cuda.empty_cache()
            
            if wandb.run is not None:    
                wandb.log({
                    "distillation/loss": loss.item(),
                    "distillation/kl_loss": kl_loss.item(),
                    "distillation/task_loss": task_loss.item(),
                    "distillation/lr": self.scheduler.get_last_lr()[0],
                })
            
        if wandb.run is not None:    
            wandb.log({
                "distillation/epoch_avg_loss": total_loss / batch_counter,
                "distillation/epoch_avg_kl_loss": total_kl_loss / batch_counter,
                "distillation/epoch_avg_task_loss": total_task_loss / batch_counter,
            })

        return {"avg_loss": total_loss / batch_counter,
            "avg_kl_loss": total_kl_loss / batch_counter,
            "avg_task_loss": total_task_loss / batch_counter,
            "total_train_tokens": total_train_tokens
        }
        
    def define_cos_scheduler_with_warmup(self, train_loader):
        """Define a cosine learning rate scheduler with warmup"""
        steps_per_epoch = len(train_loader)
        total_steps = self.config.num_epochs * steps_per_epoch

        warmup_ratio = 0.1           # 5–10% is typical
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return scheduler

    def train(self, train_loader, test_loader, ptq_args):
        """Train the student model using knowledge distillation"""            
        print(f"\nStarting knowledge distillation training...")
        print(f"Temperature: {self.config.temperature}")
        print(f"Alpha (KL weight): {ptq_args.kd_alpha}")
        print(f"KD Epochs: {ptq_args.kd_epochs}")
        self.scheduler = self.define_cos_scheduler_with_warmup(train_loader)
        self.run_eval(self.student, test_loader, utils.DEV, ptq_args, 0)
        
        for epoch in range(ptq_args.kd_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{ptq_args.kd_epochs}")
            print(f"{'='*50}")

            metrics = self.train_epoch(train_loader, ptq_args.use_distillation, ptq_args.use_alternating_LR_training)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {metrics['avg_loss']:.4f}")
            print(f"  Average KL Loss: {metrics['avg_kl_loss']:.4f}")
            print(f"  Average Task Loss: {metrics['avg_task_loss']:.4f}")
            print(f"  Total Training Tokens: {metrics['total_train_tokens']}")
            
            self.run_eval(self.student, test_loader, utils.DEV, ptq_args, epoch + 1)
            eval_utils.run_subset_benchmarks(self.student, self.tokenizer, self.device)
            
            if torch.cuda.device_count() > 1:
                dist.barrier()
                torch.cuda.empty_cache()
            

    def run_eval(self, model, test_loader, device, ptq_args, epoch=0):
        ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(self.student, test_loader, device, ptq_args) 
        print(f"Student model PPL after Epoch {epoch}: {ppl:.2f}, Avg time per token: {avg_time_per_token*1000:.2f} ms")
        
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "student_ppl": ppl
            })
    
    def save_student_model(self, save_path: str):
        """Save the distilled student model"""
        print(f"\nSaving student model to {save_path}")
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the student model"""
        self.student.eval()
        self.student.to(self.device)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.student.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                temperature=0.7,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
