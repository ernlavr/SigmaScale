from dotenv import dotenv_values, load_dotenv
import os
abs_path = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(abs_path, ".env")
load_dotenv(env_path)  # Load environment variables from
config = dotenv_values(env_path)
for key, value in config.items():
    print(f"{key}={value}")

import gc
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import math

import wandb
from utils import low_rank_utils as lru
from utils import model_utils
from train_utils.knowledge_distillation import DistillationConfig, KnowledgeDistiller

from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from modules.linears import LowRankLinear, SVDLinear
import utils.wandb_utils as wandb_utils



def decompose_weight(weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a weight matrix into low-rank factors using SVD.
    
    This provides a good initialization for training by preserving the original
    weight information as much as possible.
    
    Args:a
        weight: Original weight matrix (out_features, in_features) from nn.Linear
        rank: Target rank for decomposition
    
    Returns:
        L: (in_features, rank) matrix
        R: (rank, out_features) matrix
    """
    # Transpose to get (in_features, out_features)
    W = weight.t()
    
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # With full_matrices=False:
    # - U: (in_features, min(in_features, out_features))
    # - S: (min(in_features, out_features),)
    # - Vh: (min(in_features, out_features), out_features)
    
    # Truncate to rank
    S_truncated = S[:rank]
    U_truncated = U[:, :rank]  # (in_features, rank)
    Vh_truncated = Vh[:rank, :]  # (rank, out_features)
    
    # Distribute singular values: W ≈ (U @ sqrt(S)) @ (sqrt(S) @ V^T)
    # This is better for training than putting all S in one factor
    sqrt_S = torch.sqrt(S_truncated)
    L = U_truncated @ torch.diag(sqrt_S)  # (in_features, rank)
    R = torch.diag(sqrt_S) @ Vh_truncated  # (rank, out_features)
    
    return L, R

def run_evals_and_cleanup(model, tokenizer, device, limit=None):
    eval_utils.run_standard_benchmarks(model, tokenizer, device, limit=limit)
    gc.collect()
    torch.cuda.empty_cache()
    
def run_subset_evals_and_cleanup(model, tokenizer, device, limit=None):
    eval_utils.run_standard_benchmarks(model, tokenizer, device, limit=limit)
    

def memory_cleanup():
    print(f"Before — reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After  — reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

def get_models_data_tokenizer(model_args, training_args, ptq_args):
    print(f"Loading student model from {ptq_args.input_model}...")
    model_name = "svd_qwen" if "qwen" in ptq_args.input_model.lower() else "svd_llama"
    
    # tokenizer, data loader
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=ptq_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )
    
    train_loader = data_utils.get_train_data(ptq_args, tokenizer, mode="train")
    test_loader = data_utils.get_test_data(ptq_args, tokenizer, mode="eval")
    calib_loader = data_utils.get_calib_data(ptq_args, tokenizer)
    
    student_model = None
    try:
        student_model = model_utils.load_model(model_name, model_path=ptq_args.compressed_model)
        print(f"Successfully loaded compressed model from {ptq_args.compressed_model}")
    except Exception as e:
        print(f"Failed to load compressed model from {ptq_args.compressed_model}: {e}")
        print("Using HuggingFace AutoModelForCausalLM...")
        student_model = AutoModelForCausalLM.from_pretrained(
            ptq_args.input_model,
            device_map="auto" if torch.cuda.device_count() > 1 else "cuda",
            token=model_args.access_token,
            torch_dtype=torch.float32,  # Load in float32 for safety; can be converted later
        )
    
    print(f"Student model loaded with {sum(p.numel() for p in student_model.parameters()):,} parameters.")
    print(f"Loading teacher model from {ptq_args.input_model}...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        ptq_args.input_model,
        device_map="auto" if torch.cuda.device_count() > 1 else "cuda",
        token=model_args.access_token,
        torch_dtype=torch.float32,
    )
    
    print("Finished loading models, data, and tokenizer.")
    
    # serialize test_loader and train_loader to disk to avoid reloading every time to output_dir
    
    return student_model, teacher_model, tokenizer, test_loader, train_loader, calib_loader

def replace_linear_with_lowrank(model: nn.Module, rank_modifier: float, 
                                target_modules: Optional[list] = None,
                                init_with_svd: bool = True):
    """
    Replace Linear layers in the model with LowRankLinear.
    
    Args:
        model: The model to compress
        rank: Target rank for low-rank decomposition
        target_modules: List of exact module names to replace.
                       If None, replaces all Linear layers.
                       Examples: 
                         - ['mlp.up_proj', 'mlp.gate_proj', 'self_attn.q_proj']
                         - ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']
                       Use get_all_linear_module_names() to see available modules.
        init_with_svd: If True, initialize with SVD decomposition of original weights.
                      If False, use random initialization (for starting from scratch).
    """
    # Convert target_modules to a set for faster lookup
    target_set = set(target_modules) if target_modules is not None else None
    progress_bar = tqdm(range(0, len(target_modules)), desc="Replacing Linear with LowRankLinear", total=len(target_modules))
    
    def replace_fn(module: nn.Module, prefix: str = ''):
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}{child_name}" if prefix else child_name
            
            if isinstance(child, nn.Linear):
                # Check if this module should be replaced
                should_replace = (target_set is None) or (full_name in target_set)
                
                if should_replace:
                    in_feat = child.in_features
                    out_feat = child.out_features
                    has_bias = child.bias is not None
                    
                    # compute the rank based on modifier
                    rank = max(1, int(min(in_feat, out_feat) * rank_modifier))
                    
                    new_module = LowRankLinear(
                        in_features=in_feat,
                        out_features=out_feat,
                        rank=rank,
                        bias=has_bias,
                        device=child.weight.device,
                        dtype=child.weight.dtype
                    )
                    
                    # Initialize with SVD decomposition of original weights
                    if init_with_svd:
                        L, R = decompose_weight(child.weight, rank)
                        with torch.no_grad():
                            new_module.L.copy_(L)
                            new_module.R.copy_(R)
                            if has_bias:
                                new_module.bias.copy_(child.bias)
                    
                    setattr(module, child_name, new_module)
                    progress_bar.update(1)
            else:
                # Recursively apply to children
                replace_fn(child, prefix=f"{full_name}.")
    
    replace_fn(model)


def get_all_linear_module_names(model: nn.Module) -> list:
    """
    Get all Linear module names in the model.
    Useful for figuring out which modules to target.
    
    Returns:
        List of full module names that contain Linear layers
    """
    linear_names = []
    
    def find_linear(module: nn.Module, prefix: str = ''):
        for child_name, child in module.named_children():
            full_name = f"{prefix}{child_name}" if prefix else child_name
            
            if isinstance(child, nn.Linear):
                linear_names.append(full_name)
            else:
                find_linear(child, prefix=f"{full_name}.")
    
    find_linear(model)
    return sorted(linear_names)


def get_compression_stats(model: nn.Module) -> dict:
    """
    Calculate parameter count and compression ratio.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lowrank_params = sum(p.numel() for m in model.modules() 
                        if isinstance(m, SVDLinear) 
                        for p in m.parameters())
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'lowrank_params': lowrank_params,
    }
    
def configure_adapter(model):
    adapters.init(model)
    adapter_a_config = adapters.SeqBnConfig(
        reduction_factor=16,   # bottleneck size = hidden/factor
        non_linearity="gelu",                # activation inside the bottleneck
    )
    model.add_adapter("domain_adapter", adapter_a_config)
    model.set_active_adapters("domain_adapter")
    model.train_adapter("domain_adapter")
    
    # turn on adapter training and off all other training

    
    pass

def perform_binary_search_truncation(model, sensitivity_dict, calib_loader, args, teacher_model=None):
    return lru.binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args, teacher_model)
    
def test_calib_sensitivity_ppl(model, training_args, test_loader, model_args, ptq_args):
    return lru.calib_sensitivity_ppl(model, test_loader, ptq_args, use_cache=ptq_args.use_sensitivity_cache)
    
def process():
    memory_cleanup()
    # find_cuda_tensors()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_args, training_args, ptq_args = process_args_ptq()
    utils.set_random_seeds(seed=ptq_args.seed)
    
    student_model, teacher_model, tokenizer, test_loader, train_loader, calib_loader = get_models_data_tokenizer(model_args, training_args, ptq_args)
    student_model.seqlen = training_args.model_max_length
    
    if teacher_model is not None:
        uncompressed_stats = get_compression_stats(teacher_model)
        print(f"Compressing student with param ratio target={ptq_args.param_ratio_target}...")    
            
        
    # uncompressed_ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(student_model, test_loader, utils.DEV, ptq_args) 
    # print(f"Uncompressed PPL before low-rank replacement: {uncompressed_ppl:.2f}")
    if wandb.run is not None:
        wandb.log({
            "uncompressed/ppl": uncompressed_ppl,
            "uncompressed/avg_time_per_token": avg_time_per_token,
        })
    
    
    sensitivity = test_calib_sensitivity_ppl(student_model, training_args, test_loader,model_args, ptq_args)
    # if single gpu put on cuda
    if torch.cuda.is_available() and torch.cuda.device_count() == 1:
        student_model.to(device)
    
    if sensitivity is not None and ptq_args.apply_svd_smoothing == True:
        perform_binary_search_truncation(student_model, sensitivity, calib_loader, ptq_args, teacher_model)     
        torch.cuda.empty_cache()
        if ptq_args.save_svd_model == True:
            model_utils.save_model(student_model, model_args.input_model, model_args=model_args, training_args=training_args, ptq_args=ptq_args)
        
        compressed_ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(student_model, test_loader, utils.DEV, ptq_args)
        print(f"Scaled ppl PPL after low-rank replacement: {compressed_ppl:.2f}")
        if wandb.run is not None:
            wandb.log({
                "compressed/svd_ppl": compressed_ppl,
                "compressed/avg_time_per_token": avg_time_per_token,
            })
        
    # Print compression stats
    compressed_stats = get_compression_stats(student_model)
    print(f"\nCompression Statistics:")
    print(f"  Total parameters original: {uncompressed_stats['total_params']:,}")
    print(f"  Total parameters decomposed: {compressed_stats['total_params']:,}")
    print(f"  Low-rank parameters: {compressed_stats['lowrank_params']:,}")
    compression_ratio = compressed_stats['total_params'] / uncompressed_stats['total_params']
    print(f"  Compression ratio: {compression_ratio:.2%}")
    torch.cuda.empty_cache()
    
    # Knowledge Distillation Training
    config = DistillationConfig(
        temperature=1.0,
        alpha=ptq_args.kd_alpha,
        batch_size=2,
        learning_rate=1e-6,
        num_epochs=ptq_args.kd_epochs,
        max_seq_length=2048,
    )
    
    # Initialize distiller
    print(f"Using distillation: {ptq_args.use_distillation}")
    distiller = None
    if ptq_args.fine_tune_after_compression == True:
        distiller = KnowledgeDistiller(student_model, teacher_model, tokenizer, config, ptq_args)
        distiller.train(train_loader, test_loader, ptq_args)
        # ← free distiller BEFORE evals so its optimizer states,
    del distiller
    del train_loader
    del calib_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    
    if ptq_args.run_all_evals == True:
        run_evals_and_cleanup(student_model, tokenizer, utils.DEV)
    
    # cleanup torch memory
    del student_model
    del teacher_model
    
    del test_loader
    del tokenizer
    memory_cleanup()
    print("finished run :3")


def main():
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    
    print(sys.argv)
    
    if "--wandb_sweep" in sys.argv:
        # wandb_sweep is a parameter which has path to sweep config json file
        config_path = sys.argv[sys.argv.index("--wandb_sweep") + 1]
        wandb_utils.start_sweep(config_path, process)
    elif "--wandb_run" in sys.argv:
        wandb_utils.start_run("lowrank_replacement_test", None, process)
    else:
        process()
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dist.destroy_process_group()

def find_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(type(obj), obj.size(), obj.device, f"{obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
                
                # Find what refers to this tensor
                referrers = gc.get_referrers(obj)
                for ref in referrers:
                    if isinstance(ref, dict):
                        # Check if it's a __dict__ of some object (class/instance variables)
                        for k, v in ref.items():
                            if v is obj:
                                print(f"  -> dict key: '{k}'")
                        # Try to find which object owns this __dict__
                        owners = gc.get_referrers(ref)
                        for owner in owners:
                            if hasattr(owner, '__dict__') and owner.__dict__ is ref:
                                print(f"     owned by: {type(owner).__name__} instance")
                    elif isinstance(ref, list):
                        print(f"  -> inside a list (len={len(ref)})")
                    elif hasattr(ref, '__name__'):
                        print(f"  -> frame/function: {ref.__name__}")
        except Exception:
            pass

# Example usage for student-teacher training
if __name__ == "__main__":
    main()
    
