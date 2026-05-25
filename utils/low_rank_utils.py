import random
from typing import Tuple
import os
import json

import functools
import math
import torch
import time

import torch
from tqdm import tqdm

from utils import eval_utils, utils
# from utils.hadamard_utils import (
#     apply_exact_had_to_linear,
#     is_pow2,
#     random_hadamard_matrix,
# )
import matplotlib.pyplot as plt
from modules.linears import SVDLinear
OUTPUT_DIR = "output_dir/low_rank_analysis/singular_values"

def plot_series(data_series, title, xlabel, ylabel, output_name):
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, data in data_series.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.close()
    


def perform_svd_decomp(matrix: torch.Tensor, rank: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform SVD decomposition on the given matrix and return the low-rank approximation.

    Args:
        matrix (torch.Tensor): The input matrix to decompose.
        rank (int): The target rank for the low-rank approximation.

    Returns:
        U (torch.Tensor): Left singular vectors.
        S (torch.Tensor): Singular values.
        Vh (torch.Tensor): Right
    """
    dtype = matrix.dtype
    matrix_ = matrix.to(device="cuda", dtype=torch.float64)
    U, S, Vh = torch.linalg.svd(matrix_, full_matrices=False)
    if rank is not None:
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
    U = U.to(device="cpu", dtype=torch.float64)
    S = S.to(device="cpu", dtype=torch.float64)
    Vh = Vh.to(device="cpu", dtype=torch.float64)
    return U, S, Vh


def decompose_embeddings(model) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({"embeddings": S.detach().cpu().numpy()}, "Singular Values for Embeddings", "Index", "Singular Value", "singular_values_embeddings.png")

def decompose_attention(layer, layer_idx) -> None:
    to_plot = {}
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W, name in [(layer.self_attn.q_proj, "q_proj"), (layer.self_attn.k_proj, "k_proj"), (layer.self_attn.v_proj, "v_proj")]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        to_plot[name] = S.detach().cpu().numpy()
    
    W = layer.self_attn.o_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    to_plot["o_proj"] = S.detach().cpu().numpy()
    
    
    plot_series(to_plot, f"Layer {layer_idx}: Singular Values for Attention Output", "Index", "Singular Value", f"attention_layer_{layer_idx}.png")
    

def decompose_attention_inputs(layer, layer_idx) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W, name in [(layer.self_attn.q_proj, "q_proj"), (layer.self_attn.k_proj, "k_proj"), (layer.self_attn.v_proj, "v_proj")]:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({name: S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for Embeddings", "Index", "Singular Value", f"layer_{layer_idx}_{name}.png")


def decompose_attention_output(layer, layer_idx) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"o_proj": S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for Attention Output", "Index", "Singular Value", f"o_proj_layer_{layer_idx}.png")

def decompose_mlp(layer, layer_idx) -> None:
    # Rotate the MLP weights.
    mlp_weights = [
        (layer.mlp.up_proj, "up_proj"),
        (layer.mlp.gate_proj, "gate_proj"),
        (layer.mlp.down_proj, "down_proj"),
    ]
    to_plot = {}
    for W, name in mlp_weights:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        to_plot[name] = S.detach().cpu().numpy()
    plot_series(to_plot, f"Layer {layer_idx}: Singular Values for MLP", "Index", "Singular Value", f"mlp_layer_{layer_idx}.png")

def decompose_mlp_input(layer, layer_idx) -> None:
    # Rotate the MLP input weights.
    mlp_inputs = [(layer.mlp.up_proj, "up_proj"), (layer.mlp.gate_proj, "gate_proj")]
    for W, name in mlp_inputs:
        U, S, Vh = perform_svd_decomp(W.weight.data)
        plot_series({name: S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for MLP Input", "Index", "Singular Value", f"{name}_layer_{layer_idx}.png")

#def rotate_mlp_output(layer, R1, args):
def decompose_mlp_output(layer, layer_idx) -> None:
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"down_proj": S.detach().cpu().numpy()}, f"Layer {layer_idx}: Singular Values for MLP Output", "Index", "Singular Value", f"down_proj_layer_{layer_idx}.png")

def decompose_head(model) -> None:
    # Rotate the head.
    W = model.lm_head.weight.data
    U, S, Vh = perform_svd_decomp(W)
    plot_series({"lm_head": S.detach().cpu().numpy()}, "Singular Values for LM Head", "Index", "Singular Value", f"singular_values_lm_head.png")


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=None):
    model_id = model.config._name_or_path
    
    if use_cache is not None and os.path.exists(use_cache):
        sensitivity_dict = torch.load(use_cache, map_location="cpu")
        print(f"Loaded sensitivity results from cache: {use_cache}")
        return sensitivity_dict
    
    # get this file absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    cache_dir = os.path.join(parent_dir, "output_dir/") + "low_rank_analysis/cache"
    cache_file = f"{cache_dir}/{model_id.replace('/','_')}_calib_sensitivity_ppl.pt"
    os.makedirs(cache_dir, exist_ok=True)
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    if False: # args.compress_kv_cache
        param_ratio_candidates = [0.1 * i for i in range(1, 20)]
    else:
        param_ratio_candidates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # compute baseline ppl
    ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(model, calib_loader, utils.DEV, args)
    sensitivity_dict["baseline"] = ppl    
    
    # input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    # print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            torch.cuda.empty_cache()
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                act_aware=False,
            )
            setattr(info["father"], info["name"], svd_linear)
            ppl, avg_time_per_token = eval_utils.evaluator_single_gpu_simplified(model, calib_loader, utils.DEV, args)            
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"Processed Layer: {info['full_name']}; Param Ratio: {param_ratio}; PPL: {ppl}")
            print(f"Saving sensitivity results to: {cache_file}")
            torch.save(sensitivity_dict, cache_file)
            pbar.update(1)
            
        setattr(info["father"], info["name"], raw_linear)
        
    print(f"Saved sensitivity results to {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    return sensitivity_dict

def binary_search_truncation_rank(model, sensitivity_dict, calib_loader, args, teacher_model=None):
    """
    Binary search over truncation ranks to find optimal compression of model layers.
    
    Compresses either weight parameters or KV cache based on args.compress_kv_cache.
    Uses binary search to find compression ratios that satisfy either a perplexity
    target (ppl_target) or a parameter ratio target (ratio_target).
    
    Args:
        model: PyTorch model to compress
        sensitivity_dict: Dictionary mapping layer names to sensitivity metrics
        calib_loader: DataLoader for calibration samples
        args: Configuration object with compression parameters
        
    From: https://github.dev/hahnyuan/ASVD4LLM/blob/main/binary_search.py
    """
    # Build dictionaries for efficient module lookup    
    module_dict = {name: module for name, module in model.named_modules()}
    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    sensitivity_dict.pop("baseline", None)
    sensitivity_dict.pop("lm_head", None)
    
    if args.compress_specific_layers is not None:
        # remove all layers from layers_min_ratio that are not in compress_specific_layers
        sensitivity_dict = {k: v for k, v in sensitivity_dict.items() if int(k.split('.')[2]) in args.compress_specific_layers}
    
    if args.compress_specific_module is not None:
        # set all layers except the specific module to default_param_ratio (no compression)
        sensitivity_dict = {k:v for k, v in sensitivity_dict.items() if args.compress_specific_module in k}
    
    # DEBUG: Set args
    args.compress_kv_cache = False
    args.ppl_target = 0
    print(f"Performing binary search truncation with param ratio target {args.param_ratio_target}")
    
    
    # Recursively find all Linear modules in the model
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, torch.nn.Linear):
                full_name = full_name_dict[raw_linear]

                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    # Set compression target and filtering based on mode
    if args.compress_kv_cache:
        ratio_target = args.kv_cache_ratio_target
        # Filter sensitivity dict to only KV cache projections
        sensitivity_dict = {
            k: v for k, v in sensitivity_dict.items() 
            if "k_proj" in k or "v_proj" in k
        }
        assert args.ppl_target < 0, "ppl_target is not supported when compressing kv_cache"
        default_param_ratio = 2
    else:
        ratio_target = args.param_ratio_target
        default_param_ratio = 1

    print(
        f"=== {'compress kv_cache' if args.compress_kv_cache else 'compress weight'} target: ppl={args.ppl_target}, ratio_target={ratio_target} ==="
    )

    # Build sorted list of (layer_name, param_ratio, ppl) tuples
    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for param_ratio, ppl in v.items():
            if not args.compress_kv_cache and param_ratio >= 1:
                # we need to compress the weights, so parameter ratio should be less than 1
                continue
            sensitivity_list.append((layername, param_ratio, ppl))
    
    # Sort by perplexity (descending) - higher PPL = more sensitive layers
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # Validate that at least one target is specified
    assert args.ppl_target > 0 or ratio_target > 0
    
    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    while low < high:
        mid = (low + high) // 2
        
        # Initialize all layers with default ratio, then apply compression to sensitive ones
        layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
        
        # Compute total and compressed parameters
        tot_params = 0
        compress_params = 0
        if args.ppl_target > 0:
            # Search based on perplexity target
            assert not args.compress_kv_cache, "ppl_target is not supported when compressing kv_cache now"
            
            # Apply SVD decomposition with current ratios
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                info = linear_info[raw_linear]
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    alpha=args.alpha,
                    act_aware=args.act_aware,
                    sigma_fuse=args.sigma_fuse,
                    rank_align=args.rank_align,
                )
                setattr(info["father"], info["name"], svd_linear)
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            
            # Eval perplexity
            ppl = eval_utils.evaluator(model, calib_loader, utils.DEV, args)
            param_ratio = compress_params / tot_params
            msg = f"low={low} mid={mid}, high={high}, ppl={ppl}, param_ratio={param_ratio}"
            print(msg)
            
            # Adjust search range based on perplexity
            if ppl < args.ppl_target:
                high = mid
            else:
                low = mid + 1
        else:
            # Search based on parameter ratio target
            for layername, param_ratio in layers_min_ratio.items():
                raw_linear = module_dict[layername]
                tot_params += raw_linear.weight.numel()
                compress_params += raw_linear.weight.numel() * param_ratio
            now_ratio = compress_params / tot_params
            
            if args.compress_kv_cache:
                # because param ratio is the params for ALinear+BLienar, so the rank ratio is param ratio/2
                now_ratio /= 2
                
            msg = f"low={low} mid={mid}, high={high}, now_ratio={now_ratio}, params=({compress_params}/{tot_params})"
            print(msg)
            
            # Adjust search range based on parameter ratio
            if now_ratio > ratio_target:
                high = mid
            else:
                low = mid + 1

    print(f"=== Searching done, decomposing layers... ===")
    
    # Final decomposition with optimal ratios
    layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
    for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
        if layers_min_ratio[layername] is None:
            layers_min_ratio[layername] = param_ratio
        else:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
            
    # # Apply SVD decomposition to all layers
    # if args.train_low_rank_smoothing:
    #     activations = extract_all_layer_activations(
    #                     model, calib_loader, list(layers_min_ratio.keys())
    #                 )
        
        
    st = time.time()
    for layername, param_ratio in tqdm(layers_min_ratio.items()):
        # set ratio
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        
        # Skip decomposition if using default ratio (no compression)
        if param_ratio == default_param_ratio:
            svd_linear = raw_linear
        else:
            if args.train_low_rank_smoothing:
                activations = extract_all_layer_activations(
                            teacher_model if teacher_model is not None else model, 
                            calib_loader, 
                            list([layername])
                        )
                
                print(f"Training low-rank smoothing for layer {layername} with param ratio {param_ratio}...")
                svd_linear = SVDLinear.from_linear_with_trained_smoothing(
                    raw_linear,
                    param_ratio=param_ratio,
                    calib_data=activations[layername],
                    layer_name=layername,
                    args=args,
                )
            else:
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    # alpha=args.alpha,
                    # act_aware=args.act_aware,
                    # sigma_fuse=args.sigma_fuse,
                    # rank_align=args.rank_align,
                )
                raw_linear.to("cpu")
        
        # Replace original linear layer with SVD version
        setattr(info["father"], info["name"], svd_linear)

    ed = time.time()
    print(f"decompose time: {ed-st}")

@torch.inference_mode()
def decompose_model(model, args):
    selective_had_layers = None
    selective_had_path = getattr(args, 'selective_had_layers_path', None)

    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    decompose_embeddings(model)
    decompose_head(model)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Decomposing")):
        decompose_attention(layers[idx], idx)
        decompose_mlp(layers[idx], idx)
        
        #decompose_mlp_output(layers[idx], R1, args)
        # decompose_mlp_output(layers[idx], idx)
        # decompose_ov_proj(layers[idx], num_heads, head_dim)
        
import numpy as np


def effective_rank_participation(y, eps=1e-12):
    """
    Effective rank / participation ratio.
    
    Parameters
    ----------
    y : array-like, shape (n,)
        Non-negative values (e.g., singular values or eigenvalues).
    eps : float
        Small constant for numerical stability.
    
    Returns
    -------
    r_eff : float
        Effective rank.
    """
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 0.0)

    numerator = np.sum(y) ** 2
    denominator = np.sum(y ** 2) + eps
    return numerator / denominator


def effective_rank_entropy(y, eps=1e-12):
    """
    Entropy-based effective rank.
    
    Returns exp(H), where H is Shannon entropy.
    """
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 0.0)

    p = y / (np.sum(y) + eps)
    p = np.maximum(p, eps)

    H = -np.sum(p * np.log(p))
    return np.exp(H)





###########

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Callable
import numpy as np


class ActivationHook:
    """
    Context manager to capture intermediate layer activations during forward pass.
    
    Usage:
        hook = ActivationHook(model.layer1)
        with hook:
            output = model(input_data)
        activations = hook.activations  # (batch_size, in_features)
    """
    
    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.activations = None
        self.handle = None
    
    def _hook_fn(self, module, input, output):
        """Capture the input to the module (pre-activation)"""
        # input is a tuple, first element is the actual input
        self.activations = input[0].detach().cpu()
    
    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()


def extract_layer_activations(
    model: nn.Module,
    target_layer: nn.Module,
    dataloader: DataLoader,
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract activations (inputs) to a specific layer by running forward passes.
    
    Args:
        model: The full model
        target_layer: The specific layer whose input activations we want
        dataloader: DataLoader with calibration data (images, text tokens, etc.)
        max_batches: Max number of batches to process (None = all)
        device: Device to run on
    
    Returns:
        Activations tensor of shape (total_samples, *input_shape)
    
    Example:
        X = extract_layer_activations(
            model, 
            model.fc2,  # Get activations going INTO fc2
            calib_loader,
            max_batches=10
        )
    """
    model.eval()
    activations_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            
            # Capture activations with hook
            with ActivationHook(target_layer) as hook:
                _ = model(inputs)
                if hook.activations is not None:
                    activations_list.append(hook.activations)
    
    # Concatenate all batches
    X = torch.cat(activations_list, dim=0)
    return X


def extract_all_layer_activations(
    model: nn.Module,
    calib_loader: DataLoader,
    layer_names: list,
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Extract activations for multiple layers in a single forward pass.
    More efficient than calling extract_layer_activations separately.
    
    Args:
        model: The full model
        calib_loader: DataLoader with calibration data
        layer_names: List of (layer_name, layer_module) tuples or dict
        max_batches: Max batches to process
        device: Device to run on
    
    Returns:
        Dictionary mapping layer names to their activation tensors
    
    Example:
        activations = extract_all_layer_activations(
            model,
            calib_loader,
            layer_names=['fc1', 'fc2', 'fc3'],
            max_batches=20
        )
        X_fc1 = activations['fc1']
    """
    model.eval()
    hooks = {}
    activations_dict = {name: [] for name in layer_names}
    
    # Register hooks for all layers
    def make_hook(name):
        def hook_fn(module, input, output):
            activations_dict[name].append(input[0].detach().cpu())
        return hook_fn
    
    for name in layer_names:
        layer = dict(model.named_modules())[name]
        hooks[name] = layer.register_forward_hook(make_hook(name))
    
    try:
        with torch.no_grad():
            
            for idx, batch in tqdm(enumerate(calib_loader), desc="Extracting Activations", unit="batch"):
                batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
                batch_inputs = batch
                _ = model(batch_inputs.to(device))
    
    finally:
        # Always remove hooks
        for hook in hooks.values():
            hook.remove()
    
    # Concatenate activations for each layer
    result = {}
    for name in layer_names:
        if activations_dict[name]:
            result[name] = torch.cat(activations_dict[name], dim=0)
    
    return result


def get_layer_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Get a layer module by its name from model.named_modules().
    
    Args:
        model: PyTorch model
        layer_name: Name of layer (e.g., 'layer1.0.conv1' or just 'fc1')
    
    Returns:
        The layer module
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    raise ValueError(f"Layer '{layer_name}' not found in model")


def compute_activations_per_layer(
    model: nn.Module,
    dataloader: DataLoader,
    module_dict: Dict[str, nn.Module],
    layer_names: list,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compute activations for a list of layers. Optimized for your layer-by-layer
    SVD compression workflow.
    
    Args:
        model: The full model
        dataloader: Calibration DataLoader
        module_dict: Dictionary mapping layer names to modules (from your code)
        layer_names: List of layer names to extract activations for
        max_samples: Max total samples to collect
        device: Device to run on
    
    Returns:
        Dictionary: {layer_name: activations_tensor}
        
    Example:
        activations = compute_activations_per_layer(
            model, 
            calib_loader,
            module_dict,
            layer_names=['fc1', 'fc2'],
        )
    """
    model.to(device).eval()
    
    # Map layer names to actual modules
    layers_to_hook = {}
    for name in layer_names:
        if name in module_dict:
            layers_to_hook[name] = module_dict[name]
        else:
            # Try to get it from model
            try:
                layers_to_hook[name] = get_layer_by_name(model, name)
            except ValueError:
                print(f"Warning: Could not find layer '{name}'")
    
    # Extract activations
    activations_dict = {name: [] for name in layers_to_hook.keys()}
    total_samples = 0
    
    def make_hook(layer_name):
        def hook_fn(module, input, output):
            # Input to the linear layer is (batch_size, in_features)
            activations_dict[layer_name].append(input[0].detach().cpu())
        return hook_fn
    
    # Register hooks
    handles = {}
    for name, layer in layers_to_hook.items():
        handles[name] = layer.register_forward_hook(make_hook(name))
    
    try:
        with torch.no_grad():
            for batch in dataloader:
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                
                inputs = inputs.to(device)
                
                # Forward pass to trigger hooks
                try:
                    _ = model(inputs)
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    break
                
                total_samples += inputs.shape[0]
                
                if max_samples and total_samples >= max_samples:
                    break
    
    finally:
        # Remove all hooks
        for handle in handles.values():
            handle.remove()
    
    # Concatenate and return
    result = {}
    for name in layers_to_hook.keys():
        if activations_dict[name]:
            X = torch.cat(activations_dict[name], dim=0)
            result[name] = X
            print(f"Layer '{name}': Extracted {X.shape[0]} samples, shape {X.shape}")
        else:
            print(f"Warning: No activations collected for layer '{name}'")
    
    return result


# ============================================================================
# INTEGRATION: Modified version of your SVD loop with activation computation
# ============================================================================

def apply_svd_to_layers_with_activations(
    model: nn.Module,
    module_dict: Dict[str, nn.Module],
    linear_info: Dict,
    layers_min_ratio: Dict[str, float],
    calib_loader: DataLoader,
    SVDLinear,
    SVDLinear_Smoothed,
    default_param_ratio: float = 1.0,
    train_low_rank_smoothing: bool = False,
    device: str = "cuda",
    max_calib_batches: Optional[int] = None,
):
    """
    Complete SVD decomposition pipeline with activation extraction.
    
    This wraps your existing SVD loop and adds activation computation.
    
    Args:
        model: The full model
        module_dict: Dict mapping layer names to their modules
        linear_info: Info dict for each layer
        layers_min_ratio: Dict mapping layer names to compression ratios
        calib_loader: Calibration data loader
        SVDLinear: Your SVD layer class
        SVDLinear_Smoothed: Your smoothed SVD layer class
        default_param_ratio: Ratio for no compression
        train_low_rank_smoothing: Whether to train smooth factors
        device: Device to use
        max_calib_batches: Limit calibration batches for speed
    """
    import time
    from tqdm import tqdm
    
    model.to(device).eval()
    
    # STEP 1: Extract activations for all layers at once
    print("=" * 60)
    print("STEP 1: Extracting activations for all layers...")
    print("=" * 60)
    
    layer_names = list(layers_min_ratio.keys())
    activations = compute_activations_per_layer(
        model,
        calib_loader,
        module_dict,
        layer_names,
        max_samples=None,
        device=device,
    )
    
    print(f"\nExtracted activations for {len(activations)} layers\n")
    
    # STEP 2: Apply SVD decomposition to each layer
    print("=" * 60)
    print("STEP 2: Applying SVD decomposition to layers...")
    print("=" * 60)
    
    st = time.time()
    
    for layername, param_ratio in tqdm(layers_min_ratio.items()):
        raw_linear = module_dict[layername]
        info = linear_info[raw_linear]
        
        # Skip if using default ratio
        if param_ratio == default_param_ratio:
            svd_linear = raw_linear
        else:
            # Get activations for this layer
            X = activations.get(layername)
            
            if X is None:
                print(f"Warning: No activations for {layername}, skipping")
                svd_linear = raw_linear
            else:
                print(f"\n{layername}:")
                print(f"  Activations shape: {X.shape}")
                print(f"  Layer weight shape: {raw_linear.weight.shape}")
                print(f"  Compression ratio: {param_ratio}")
                
                if train_low_rank_smoothing:
                    svd_linear = SVDLinear_Smoothed.from_linear_with_trained_smoothing(
                        raw_linear,
                        param_ratio=param_ratio,
                        calib_data=X,  # Pass extracted activations
                    )
                else:
                    svd_linear = SVDLinear.from_linear(
                        raw_linear,
                        param_ratio=param_ratio,
                        # act_aware=True,  # Can use activations here
                        # calib_data=X,
                    )
        
        raw_linear.to("cpu")
        
        # Replace original layer with SVD version
        setattr(info["father"], info["name"], svd_linear)
    
    ed = time.time()
    print(f"\n{'=' * 60}")
    print(f"Total decomposition time: {ed - st:.2f}s")
    print(f"{'=' * 60}")
    
    return model


# ============================================================================
# MEMORY-EFFICIENT VARIANT: For very large models
# ============================================================================

def extract_activations_streaming(
    model: nn.Module,
    target_layer: nn.Module,
    dataloader: DataLoader,
    chunk_size: int = 1000,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract activations in chunks to save memory for large datasets.
    Returns activations one chunk at a time for processing.
    
    Args:
        model: The full model
        target_layer: Layer whose inputs we're capturing
        dataloader: Calibration DataLoader
        chunk_size: Number of samples to keep in memory per chunk
        device: Device to run on
    
    Yields:
        Chunks of activations of size (chunk_size, input_features)
    """
    model.eval()
    buffer = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            
            with ActivationHook(target_layer) as hook:
                _ = model(inputs)
                if hook.activations is not None:
                    buffer.append(hook.activations)
            
            # Yield when buffer is full
            if sum(x.shape[0] for x in buffer) >= chunk_size:
                yield torch.cat(buffer, dim=0)
                buffer = []
    
    # Yield remaining
    if buffer:
        yield torch.cat(buffer, dim=0)