import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def effective_rank_participation(y, eps=1e-12):
    y = torch.as_tensor(y, dtype=torch.float32)
    y = torch.clamp(y, min=0.0)
    numerator = torch.sum(y) ** 2
    denominator = torch.sum(y ** 2) + eps
    return numerator / denominator


def effective_rank_entropy(y, eps=1e-12):
    y = torch.clamp(y, min=0.0)
    p = y / (torch.sum(y) + eps)
    p = torch.clamp(p, min=eps)
    H = -torch.sum(p * torch.log(p))
    return torch.exp(H)

def print_plot_stats(singular_values, rank, loss, plot_name, output_dir="output_dir/singular_values_smoother/", first_sigmas=None, epoch=None):
    """ Print effective rank statistics for the given singular values. """
    S_detached = singular_values
    entropy = effective_rank_entropy(S_detached).item()
    participation = effective_rank_participation(S_detached).item()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.plot(S_detached.detach().cpu().numpy(), label=f'Scaled Singular Values', alpha=0.75)
    # Optional overlay
    if first_sigmas is not None:
        plt.plot(
            first_sigmas,
            linestyle="--",
            alpha=0.75,
            label="Initial Singular Values"
        )

    plt.axvline(x=rank, color='r', linestyle='--', label=f'Rank {rank}')
    plt.title(f'Singular Values - {plot_name}; \nEntropy: {entropy:.2f}, Particip: {participation:.2f}, Loss: {loss:.4f}')
    plt.xlabel('Index')
    plt.ylabel('Sigma Magnitude')
    plt.legend()
    plt.grid()
    plt.savefig(output_dir + f"{plot_name}_singular_values.png")
    # change to log scale for better visibility
    plt.xscale('log')
    plt.savefig(output_dir + f"{plot_name}_singular_values_xlog.png")
    plt.close()
    return entropy, participation
    
    
    
def truncate_svd(U, S, V, rank):
    """ Truncate the SVD components to the specified rank. """
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    V_truncated = V[:, :rank]
    return U_truncated, S_truncated, V_truncated


def train_svd_smoother(X, calib_data, rank, num_epochs=100, lr=1e-3):
    """
    Train a smoothing function for better low-rank decomposition.
    
    Args:
        X: The original weight matrix (torch.Tensor).
        calib_data: Calibration data (torch.Tensor) to evaluate the approximation quality.
        rank: The target rank for the low-rank approximation.
        num_epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
    """
    
    # Initialize a weight matrix for the smoother
    smoother = nn.Parameter(torch.randn(X.shape[0]))
    optimizer = torch.optim.Adam([smoother], lr=lr)
    
    # put smoother on the same device as X
    smoother = smoother.to(X.device)
    
    # Compute the low-rank approximation of the smoothed weights
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        S_prime = torch.diag(S) @ torch.diag(smoother)
        
        U_truncated, S_truncated, V_truncated = truncate_svd(U, S_prime, V, rank)
        
        w_approx = U_truncated @ S_truncated @ V_truncated
        
        # Compute the loss based on the calibration data
        loss = loss_fn(w_approx, X, mode="frobenius")  # Example loss, can be replaced with a more relevant one
        
        
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    
    
    
def loss_fn_wprime(W, W_prime, X, mode="frobenius"):
    # Compute activations through original and compressed weights
    original_output = W @ X
    compressed_output = W_prime @ X
    
    # Frobenius norm of the difference
    diff = original_output - compressed_output
    loss = torch.norm(diff, p='fro') ** 2
    
    # Normalize by matrix size for stability
    loss = loss / (W.shape[0] * W.shape[1])
    return loss
    
    
def loss_fn(W, USV, X, rank, mode="frobenius"):
    """
    Compute compression loss: ||W @ X - W' @ X||_F
    
    Args:
        W: Original weight matrix (d_out, d_in)
        W_prime: Compressed weight matrix (d_out, d_in)
        X: Activation data (d_in, batch_size) or (batch_size, d_in)
        mode: Loss computation mode
    
    Returns:
        Scalar loss value
    """
    U, S, V = USV
    if mode == "frobenius":
        # Compute activations through original and compressed weights
        U_trunc, S_trunc, V_trunc = truncate_svd(U, S, V, rank)
                
        # Reconstruct compressed weight matrix
        L, R = reconstruct_from_svd(U_trunc, S_trunc, V_trunc)
        LR = (L, R)  # Store as tuple for loss function
        # Compute loss: ||W @ X - W' @ X||_F
        original_output = W @ X
        compressed_output = L @ (R @ X)
        
        # Frobenius norm of the difference
        diff = original_output - compressed_output
        loss = torch.norm(diff, p='fro') ** 2
        
        # Normalize by matrix size for stability
        loss = loss / (W.shape[0] * W.shape[1])
        return loss
    
    if mode == "min_S":
        # minimize entropy of singular values
        entropy = effective_rank_entropy(S.detach().cpu().numpy())
        
        pass    
    
    elif mode == "calibration":
        raise NotImplementedError("Calibration-based loss not implemented yet")
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def truncate_svd(U, S, V, rank):
    """
    Truncate SVD components to the specified rank.
    
    Args:
        U: Left singular vectors (d_out, min(d_out, d_in))
        S: Singular values (min(d_out, d_in),)
        V: Right singular vectors (d_in, min(d_out, d_in))
        rank: Target rank
    
    Returns:
        Truncated components for low-rank reconstruction
    """
    U_truncated = U[:, :rank]
    S_truncated = S[:rank]
    V_truncated = V[:rank, :]  # Note: torch.linalg.svd returns V as (d_in, min(d_out, d_in))
    
    return U_truncated, S_truncated, V_truncated


def reconstruct_from_svd(U, S, V):
    """
    Reconstruct matrix from SVD components: W = U @ diag(S) @ V
    
    Args:
        U: Left singular vectors
        S: Singular values (1D vector)
        V: Right singular vectors
    
    Returns:
        Reconstructed matrix
    """
    # constrtuct L-R components
    L = U @ torch.sqrt(torch.diag(S))
    R = torch.sqrt(torch.diag(S)) @ V
    return L, R

def intercept_sanitize_nans(d_r, d_c):
    num_sant = 0
    for param in [d_r, d_c]:
        # Check if the parameter exists in the current scaling mode and has a gradient
        if param is not None and param.grad is not None:
            invalid_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
            
            if invalid_mask.any():
                valid_mask = ~invalid_mask
                
                # Calculate the mean of the healthy gradients
                if valid_mask.any():
                    replacement_val = param.grad[valid_mask].mean()
                else:
                    # Fallback if the ENTIRE gradient tensor blew up
                    replacement_val = torch.tensor(0.0, device=param.device)
                
                # Overwrite the NaNs/Infs with the replacement value
                param.grad = torch.where(invalid_mask, replacement_val, param.grad)
                
                # log number of sanitized gradients
                num_sant += invalid_mask.sum().item()
    
    return num_sant    
                    
        # -----------------------------------------------------

def get_scheduler(optimizer, num_epochs, args):
    scheduler_name = args.scheduler_name
    scheduler = None
    match scheduler_name:
        case "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        case "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        case "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            
        case "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5)
            
        case "linear_cosine":
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=50)
            cosine = CosineAnnealingLR(optimizer, T_max=num_epochs-50)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[50])
        case "none" | None:
            scheduler = None
            
    return scheduler

def train_svd_compressor_alternating_optimization(W, X, rank, args, num_epochs=40, lr=1e-2, device=None):
    """
    Alternate optimization of low-rank factors L and R for weight compression.
    
    This optimizes L and R in an alternating fashion to minimize reconstruction error
    on the given activation data X.
    
    Args:
        W: Original weight matrix (torch.Tensor, shape: d_out x d_in)
        X: Activation data (torch.Tensor, shape: d_in x batch_size or batch_size x d_in)
        rank: Target rank for compression
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to place tensors on
    Returns:
        L, R: Optimized low-rank factors such that W_approx = L @ R
        loss_history: List of loss values during training
    """

def train_svd_compressor(W, X, rank, args, num_epochs=40, lr=1e-2, layername=None, device=None):
    """
    Optimize weight compression using learnable singular value scaling.
    
    This learns a per-singular-value scaling factor to minimize reconstruction error
    on the given activation data X.
    
    Args:
        W: Original weight matrix (torch.Tensor, shape: d_out x d_in)
        X: Activation data (torch.Tensor, shape: d_in x batch_size or batch_size x d_in)
        rank: Target rank for compression
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to place tensors on
    
    Returns:
        W_compressed: Optimized compressed weight matrix
        loss_history: List of loss values during training
    """
    if device is None:
        device = W.device
    
    W = W.to(device)
    X = X.to(device)
    # squeeze out single dimnesions
    batch, tokens, d_in = X.shape
    X = X.reshape(-1, d_in)
    
    
    
    # Ensure X is in the right shape (d_in x batch_size)
    if X.shape[0] != W.shape[1]:
        if X.shape[1] == W.shape[1]:
            X = X.T
        else:
            raise ValueError(
                f"X shape {X.shape} incompatible with W shape {W.shape}. "
                f"X should have {W.shape[1]} features."
            )
    # Compute no smooth loss for reference
    W_no_smooth = W
    U_ns, S_ns, V_ns = torch.linalg.svd(W_no_smooth, full_matrices=False)
    USV_ns = U_ns, S_ns, V_ns
    
    loss_no_smooth = loss_fn(W, USV_ns, X, rank, mode="frobenius")
    entr, particip = print_plot_stats(S_ns, rank, plot_name="No_Smooth")
    print(f"Truncated Loss: {loss_no_smooth.item():.6f}; Entropy: {entr:.2f}, Participation: {particip:.2f}")
    
    if wandb.run is not None:
        wandb.log({
            "baseline/loss": loss_no_smooth.item(),
            "baseline/entropy_no_smooth": entr,
            "baseline/participation_no_smooth": particip,
        })
        
    # Compute statistics of W for smart initialization
    w_std = W.std().item()

    # Initialize learnable scaling vectors with scale proportional to W
    # Keep them close to 0 in log-space (exp(0) = 1, meaning neutral scaling initially)
    d_r = nn.Parameter(torch.randn(W.shape[0], device=W.device, dtype=W.dtype) * w_std * 0.1)
    d_c = nn.Parameter(torch.randn(W.shape[1], device=W.device, dtype=W.dtype) * w_std * 0.1)
    
    
    lr = args.learning_rate
    scaling = args.scaling_algo
    l2_regularizer_scale = args.l2_regularizer_scale
    regularizing_noise = args.add_regularizing_noise
    intercept_sanitize_grad = args.intercept_sanitize_grad
    add_term_to_loss = args.add_term_to_loss
    add_term_loss_scaler = args.term_loss_scaler
    num_sanitized = 0
    print(f"Training SVD smoother with scaling: {scaling}, lr: {lr}, l2_regularizer_scale: {l2_regularizer_scale}, regularizing_noise: {regularizing_noise}, intercept_sanitize_grad: {intercept_sanitize_grad}")
    
    optimizer = None
    if scaling == "cols":
        optimizer = torch.optim.AdamW([d_c], lr=lr, weight_decay=l2_regularizer_scale)
    elif scaling == "rows":
        optimizer = torch.optim.AdamW([d_r], lr=lr, weight_decay=l2_regularizer_scale)
    elif scaling == "both":
        optimizer = torch.optim.AdamW([d_r, d_c], lr=lr, weight_decay=l2_regularizer_scale)
    
    print(optimizer)
    scheduler = get_scheduler(optimizer, num_epochs, args)
    
    loss_history = []
    gradient_norms_d_r = []
    gradient_norms_d_c = []
    time_started = time.time()
    times = []
    output_dir = "output_dir/singular_values_smoother/hadamard_100/"
    for epoch in tqdm(range(num_epochs)):
        # SCALING
        if scaling == "cols":
            D_c = torch.diag(torch.exp(d_c))
            D_c_inv = torch.diag(torch.exp(-d_c))
        elif scaling == "rows":
            D_r = torch.diag(torch.exp(d_r))
            D_r_inv = torch.diag(torch.exp(-d_r))  # Inverse is straightforward with exp    
        elif scaling == "both":
            D_r = torch.diag(torch.exp(d_r))
            D_c = torch.diag(torch.exp(d_c))
            D_r_inv = torch.diag(torch.exp(-d_r))
            D_c_inv = torch.diag(torch.exp(-d_c))
        
        
        
        # Check diagonal values before SVD
        print(f"Epoch {epoch}")
        print(f"  exp(d_r) range: [{torch.exp(d_r).min():.4f}, {torch.exp(d_r).max():.4f}]; mean: {torch.exp(d_r).mean():.4f}; std: {torch.exp(d_r).std():.4f}; range: {torch.exp(d_r).max()}, {torch.exp(d_r).min():.4f}")
        print(f"  exp(d_c) range: [{torch.exp(d_c).min():.4f}, {torch.exp(d_c).max():.4f}]; mean: {torch.exp(d_c).mean():.4f}; std: {torch.exp(d_c).std():.4f}; range: {torch.exp(d_c).max()}, {torch.exp(d_c).min():.4f}")
        
        # Apply row/column scaling
        if scaling == "cols":
            W_scaled = W @ D_c
        elif scaling == "rows":
            W_scaled = D_r @ W
        elif scaling == "both":
            W_scaled = D_r @ W @ D_c
            
        # FIX: Inject tiny noise to prevent SVD degenerate singular values
        if regularizing_noise is not None:
            W_scaled = W_scaled + torch.randn_like(W_scaled) * regularizing_noise
        
        print(f"  W_scaled has broken? {torch.isnan(W_scaled).any()}")
        
        # SVD in scaled space
        U_scaled, S_scaled, V_scaled = torch.linalg.svd(W_scaled, full_matrices=False)
        
        # Truncate in scaled space
        U_trunc, S_trunc, V_trunc = truncate_svd(U_scaled, S_scaled, V_scaled, rank)
        
        # Recover original space
        W_trunc_scaled = U_trunc @ torch.diag(S_trunc) @ V_trunc
        
        if scaling == "cols":
            W_trunc = W_trunc_scaled @ D_c_inv  # Only column scaling, so recover with D_c_inv
        elif scaling == "rows":
            W_trunc = D_r_inv @ W_trunc_scaled  # Only row scaling, so recover with D_r_inv
        elif scaling == "both":
            W_trunc = D_r_inv @ W_trunc_scaled @ D_c_inv
        
        
        optimizer.zero_grad()
        loss = loss_fn_wprime(W, W_trunc, X, mode="frobenius")
    
        if torch.isnan(loss):
            print("NaN detected in loss! Stopping.")
            break
        
        # add regularization to loss to keep scaling factors from diverging
        if args.optimizer_constraint == "l2" or args.optimizer_constraint == "all":    
            # L2 regularization to keep parameters from growing too large
            if scaling == "cols":
                norm = l2_regularizer_scale * torch.norm(d_c) ** 2
            elif scaling == "rows":
                norm = l2_regularizer_scale * torch.norm(d_r) ** 2
            elif scaling == "both":
                norm = l2_regularizer_scale * (torch.norm(d_r) ** 2 + torch.norm(d_c) ** 2)
            loss = loss + norm
        
        # if adding  entropy minimization, use scaling 1e-4 to 1e-6
        entropy = None
        participation = None
        loss_w_term = None
        if add_term_to_loss == 'entropy':
            entropy = effective_rank_entropy(S_trunc) * add_term_loss_scaler
            loss_w_term = loss + entropy
        elif add_term_to_loss == 'participation':
            participation = effective_rank_participation(S_trunc) * add_term_loss_scaler
            loss_w_term = loss + participation
        
        if add_term_to_loss is not None:
            loss_w_term.backward()
        else:
            loss.backward()
        
        # sanitize gradients for intercept term if enabled
        if intercept_sanitize_grad:
            n = intercept_sanitize_nans(d_r, d_c)
            if n > 0:
                num_sanitized += 1  # This will be updated by the sanitize function
        optimizer.step()
        
        
        
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step()
            
        loss_history.append(loss.item())
        
        if scaling == "cols":
            gradient_norms_d_c.append(d_c.grad.norm().item())
        elif scaling == "rows":
            gradient_norms_d_r.append(d_r.grad.norm().item())
        elif scaling == "both":
            gradient_norms_d_r.append(d_r.grad.norm().item())
            gradient_norms_d_c.append(d_c.grad.norm().item())
        
        time_elapsed = time.time() - time_started
        time_started = time.time()
        times.append(time_elapsed)
        
        
        # log loss, entropy, participation, mean+std of smooth operator
        if wandb.run is not None:
            log_dict = {
                f"{layername}/loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else lr,
            }
            
            if add_term_to_loss is not None:
                log_dict[f"{layername}/loss_with_term"] = loss_w_term.item()
            
            if num_sanitized is not None:
                log_dict[f"{layername}/num_sanitized"] = num_sanitized
                
            if entropy is not None:
                log_dict[f"{layername}/entropy"] = entropy.item()
            
            if participation is not None:
                log_dict[f"{layername}/participation"] = participation.item()
            
            if loss_w_term is not None:
                log_dict[f"{layername}/loss_with_term"] = loss_w_term.item()

            wandb.log(log_dict)
        
        # Clear cache periodically
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            recent_loss = sum(loss_history[-10:]) / 10
            mean_time = sum(times[-10:]) / 10
            entr, particip = print_plot_stats(S_scaled, rank, plot_name=f"Learned_Scaling_Epoch{epoch}", output_dir=output_dir)
            
            if scaling == "cols":
                gradient_norms_c = np.array(gradient_norms_d_c[-10:])
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}; Avg (last 10): {recent_loss:.6f}; Time: {mean_time:.4f}s; Entropy: {entr:.2f}, Participation: {particip:.2f}; Gradient norm d_c={gradient_norms_c.mean():.4f}; Gradient min-max d_c=({gradient_norms_c.min():.4f}, {gradient_norms_c.max():.4f})")
            elif scaling == "rows":
                gradient_norms_r = np.array(gradient_norms_d_r[-10:])
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}; Avg (last 10): {recent_loss:.6f}; Time: {mean_time:.4f}s; Entropy: {entr:.2f}, Participation: {particip:.2f}; Gradient norm d_r={gradient_norms_r.mean():.4f}")
            elif scaling == "both":
                gradient_norms_r = np.array(gradient_norms_d_r[-10:])
                gradient_norms_c = np.array(gradient_norms_d_c[-10:])
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.6f}; Avg (last 10): {recent_loss:.6f}; Time: {mean_time:.4f}s; Entropy: {entr:.2f}, Participation: {particip:.2f}; Gradient norms: d_r={gradient_norms_r.mean():.4f}, d_c={gradient_norms_c.mean():.4f}")
                
        early_stopping_threshold = 1e-4
        # if last 100 haven't improved by more than threshold, stop, or if loss is growing, stop
        if len(loss_history) > 100:
            recent_avg = sum(loss_history[-50:]) / 50
            prev_avg = sum(loss_history[-100:-50]) / 50
            if abs(prev_avg - recent_avg) < early_stopping_threshold:
                print(f"Early stopping at epoch {epoch} due to minimal improvement over last 100 epochs.")
                break
            if recent_avg > prev_avg:
                print(f"Early stopping at epoch {epoch} due to loss increase over last 100 epochs.")
                break        
            
    # # Final compressed weight matrix with optimized scaling
    # with torch.no_grad():
    #     S_final = S_trunc * smooth_operator[:rank]  # Apply learned scaling to singular values
    #     W_compressed = reconstruct_from_svd(U_trunc, S_final, V_trunc)
        
    
    with torch.no_grad():
        if scaling == "cols":
            W_final_scaled = W @ D_c
        elif scaling == "rows":
            W_final_scaled = D_r @ W
        elif scaling == "both":
            W_final_scaled = D_r @ W @ D_c
            
        U_final, S_final, V_final = torch.linalg.svd(W_final_scaled, full_matrices=False)
        U_trunc, S_trunc, V_trunc = truncate_svd(U_final, S_final, V_final, rank)
        
        # Compute L and R components by splitting singular values
        sqrt_S = torch.sqrt(torch.diag(S_trunc))
        L_scaled = U_trunc @ sqrt_S  # shape: (d_out, rank)
        R_scaled = sqrt_S @ V_trunc  # shape: (rank, d_in)
        
        # Recover to original space based on scaling mode
        if scaling == "cols":
            L = L_scaled  # No row scaling applied
            R = R_scaled @ D_c_inv
        elif scaling == "rows":
            L = D_r_inv @ L_scaled
            R = R_scaled  # No column scaling applied
        elif scaling == "both":
            L = D_r_inv @ L_scaled
            R = R_scaled @ D_c_inv
        
        # Compute W_compressed for reference
        W_compressed = L @ R
        # crosscheck loss again
        final_loss = loss_fn_wprime(W, W_compressed, X, mode="frobenius")
        print(f"Final Loss after training smoother: {final_loss.item():.6f}")

    return L, R, loss_history


def compute_compression_ratio(W, rank):
    """
    Compute the compression ratio when using rank-r SVD approximation.
    
    Original storage: d_out * d_in
    Compressed storage: d_out * r + r + r * d_in
    
    Args:
        W: Weight matrix
        rank: Target rank
    
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    d_out, d_in = W.shape
    original_params = d_out * d_in
    compressed_params = d_out * rank + rank + rank * d_in
    return original_params / compressed_params



















def _run_svd_training_phase(
    W, X, rank, args,
    d_r, d_c,
    params_to_train,   # list of nn.Parameter objects to optimize in this phase
    scaling,           # "rows" | "cols" | "both"
    num_epochs=40,
    lr=1e-2,
    phase_name="",
):
    """
    Shared inner training loop used by both simultaneous and alternating optimizers.

    Args:
        W:               Original weight matrix (d_out x d_in)
        X:               Activation data, already transposed to (d_in x N)
        rank:            Target rank
        args:            Argument namespace (same as callers receive)
        d_r:             Row-scaling nn.Parameter  (d_out,)
        d_c:             Col-scaling nn.Parameter  (d_in,)
        params_to_train: Which of [d_r, d_c] to pass to the optimizer
        scaling:         Which scaling matrices are active
        num_epochs:      Epochs to run
        lr:              Base learning rate (overridden by args.learning_rate)
        phase_name:      String prefix used in log / print messages

    Returns:
        loss_history: list of per-epoch loss values
    """
    lr                    = args.learning_rate
    l2_regularizer_scale  = args.l2_regularizer_scale
    regularizing_noise    = args.add_regularizing_noise
    intercept_sanitize_grad = args.intercept_sanitize_grad
    add_term_to_loss      = args.add_term_to_loss
    add_term_loss_scaler  = args.term_loss_scaler

    prefix = f"[{phase_name}] " if phase_name else ""
    print(
        f"{prefix}Training SVD smoother | scaling={scaling}, lr={lr}, "
        f"l2={l2_regularizer_scale}, noise={regularizing_noise}, "
        f"sanitize_grad={intercept_sanitize_grad}"
    )

    optimizer = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=l2_regularizer_scale)
    print(optimizer)
    scheduler = get_scheduler(optimizer, num_epochs, args)

    loss_history        = []
    gradient_norms_d_r  = []
    gradient_norms_d_c  = []
    num_sanitized       = 0
    time_started        = time.time()
    times               = []
    output_dir          = "output_dir/singular_values_smoother/hadamard_100/"
    first_sigmas = None

    for epoch in tqdm(range(num_epochs)):
        if first_sigmas is None:
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            first_sigmas = S.detach().cpu().numpy()
            
        # ------------------------------------------------------------------ #
        # Build scaling matrices from current parameters                      #
        # ------------------------------------------------------------------ #
        D_r, D_c, D_r_inv, D_c_inv = None, None, None, None
        if scaling in ("rows", "both"):
            D_r     = torch.diag(torch.exp(d_r))
            D_r_inv = torch.diag(torch.exp(-d_r))
        if scaling in ("cols", "both"):
            D_c     = torch.diag(torch.exp(d_c))
            D_c_inv = torch.diag(torch.exp(-d_c))

        print(f"{prefix}Epoch {epoch}")
        if D_r is not None:
            print(
                f"  exp(d_r) range: [{torch.exp(d_r).min():.4f}, {torch.exp(d_r).max():.4f}]"
                f"  mean: {torch.exp(d_r).mean():.4f}  std: {torch.exp(d_r).std():.4f}"
            )
        if D_c is not None:
            print(
                f"  exp(d_c) range: [{torch.exp(d_c).min():.4f}, {torch.exp(d_c).max():.4f}]"
                f"  mean: {torch.exp(d_c).mean():.4f}  std: {torch.exp(d_c).std():.4f}"
            )
            
        # ------------------------------------------------------------------ #
        # Apply scaling, optional noise, SVD + truncation                     #
        # ------------------------------------------------------------------ #
        if scaling == "cols":
            W_scaled = W @ D_c
        elif scaling == "rows":
            W_scaled = D_r @ W
        else:  # both
            W_scaled = D_r @ W @ D_c

        if regularizing_noise is not None:
            W_scaled = W_scaled + torch.randn_like(W_scaled) * regularizing_noise

        print(f"  W_scaled has NaN? {torch.isnan(W_scaled).any()}")

        U_scaled, S_scaled, V_scaled = torch.linalg.svd(W_scaled, full_matrices=False)
        U_trunc, S_trunc, V_trunc    = truncate_svd(U_scaled, S_scaled, V_scaled, rank)
        W_trunc_scaled               = U_trunc @ torch.diag(S_trunc) @ V_trunc
        

        if scaling == "cols":
            W_trunc = W_trunc_scaled @ D_c_inv
        elif scaling == "rows":
            W_trunc = D_r_inv @ W_trunc_scaled
        else:  # both
            W_trunc = D_r_inv @ W_trunc_scaled @ D_c_inv

        # ------------------------------------------------------------------ #
        # Loss + optional auxiliary terms                                     #
        # ------------------------------------------------------------------ #
        optimizer.zero_grad()
        loss = loss_fn_wprime(W, W_trunc, X, mode="frobenius")

        if torch.isnan(loss):
            print(f"{prefix}NaN detected in loss! Stopping.")
            break

        if args.optimizer_constraint in ("l2", "all"):
            if scaling == "cols":
                norm = l2_regularizer_scale * torch.norm(d_c) ** 2
            elif scaling == "rows":
                norm = l2_regularizer_scale * torch.norm(d_r) ** 2
            else:
                norm = l2_regularizer_scale * (torch.norm(d_r) ** 2 + torch.norm(d_c) ** 2)
            loss = loss + norm

        entropy, participation, loss_w_term = None, None, None
        if add_term_to_loss == "entropy":
            entropy      = effective_rank_entropy(S_trunc) * add_term_loss_scaler
            loss_w_term  = loss + entropy
        elif add_term_to_loss == "participation":
            participation = effective_rank_participation(S_trunc) * add_term_loss_scaler
            loss_w_term   = loss + participation

        (loss_w_term if loss_w_term is not None else loss).backward()

        if intercept_sanitize_grad:
            n = intercept_sanitize_nans(d_r, d_c)
            if n > 0:
                num_sanitized += 1

        optimizer.step()

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(loss)
            else:
                scheduler.step()

        # ------------------------------------------------------------------ #
        # Bookkeeping                                                         #
        # ------------------------------------------------------------------ #
        loss_history.append(loss.item())

        if d_r.grad is not None:
            gradient_norms_d_r.append(d_r.grad.norm().item())
        if d_c.grad is not None:
            gradient_norms_d_c.append(d_c.grad.norm().item())

        time_elapsed = time.time() - time_started
        time_started = time.time()
        times.append(time_elapsed)

        if wandb.run is not None:
            log_dict = {
                f"{phase_name}/loss": loss.item(),
                f"{phase_name}/learning_rate": (
                    scheduler.get_last_lr()[0] if scheduler is not None else lr
                ),
            }
            if loss_w_term  is not None: log_dict[f"{phase_name}/loss_with_term"] = loss_w_term.item()
            if num_sanitized:            log_dict[f"{phase_name}/num_sanitized"]  = num_sanitized
            if entropy      is not None: log_dict[f"{phase_name}/entropy"]        = entropy.item()
            if participation is not None:log_dict[f"{phase_name}/participation"]  = participation.item()
            wandb.log(log_dict)

        torch.cuda.empty_cache()

        if (epoch + 1) % 50 == 0:
            recent_loss = sum(loss_history[-10:]) / max(len(loss_history[-10:]), 1)
            mean_time   = sum(times[-10:])        / max(len(times[-10:]),        1)
            entr, particip = print_plot_stats(
                S_scaled, rank, loss.item(),
                plot_name=f"{phase_name}_Epoch{epoch}",
                output_dir=output_dir,
                first_sigmas=first_sigmas,
            )
            gnr = np.array(gradient_norms_d_r[-10:]) if gradient_norms_d_r else np.zeros(1)
            gnc = np.array(gradient_norms_d_c[-10:]) if gradient_norms_d_c else np.zeros(1)
            print(
                f"{prefix}Epoch {epoch:3d} | Loss: {loss.item():.6f} | "
                f"Avg10: {recent_loss:.6f} | Time: {mean_time:.4f}s | "
                f"Entropy: {entr:.2f} | Participation: {particip:.2f} | "
                f"grad d_r={gnr.mean():.4f} | grad d_c={gnc.mean():.4f}"
            )

        # Early stopping
        # if len(loss_history) > 100:
        #     recent_avg = sum(loss_history[-50:])   / 50
        #     prev_avg   = sum(loss_history[-100:-50]) / 50
        #     if abs(prev_avg - recent_avg) < 1e-4:
        #         print(f"{prefix}Early stopping at epoch {epoch}: minimal improvement.")
        #         break
        #     if recent_avg > prev_avg:
        #         print(f"{prefix}Early stopping at epoch {epoch}: loss is increasing.")
        #         break

    return loss_history


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def _init_parameters_and_baseline(W, X, rank, args, device):
    """
    Shared setup: move tensors, reshape X, print baseline SVD loss, and
    initialise d_r / d_c parameters.

    Returns:
        W, X, d_r, d_c   (all on device, X in shape d_in x N)
    """
    W = W.to(device)
    X = X.to(device)
    X = X.squeeze()
    batch, tokens, d_in = X.shape
    X = X.reshape(-1, d_in)

    if X.shape[0] != W.shape[1]:
        if X.shape[1] == W.shape[1]:
            X = X.T
        else:
            raise ValueError(
                f"X shape {X.shape} incompatible with W shape {W.shape}. "
                f"Expected {W.shape[1]} features."
            )

    # Baseline (no smoothing)
    U_ns, S_ns, V_ns = torch.linalg.svd(W, full_matrices=False)
    loss_no_smooth    = loss_fn(W, (U_ns, S_ns, V_ns), X, rank, mode="frobenius")
    entr, particip    = print_plot_stats(S_ns, rank, loss_no_smooth.item(), plot_name="No_Smooth")
    print(
        f"Truncated Loss: {loss_no_smooth.item():.6f}; "
        f"Entropy: {entr:.2f}, Participation: {particip:.2f}"
    )
    if wandb.run is not None:
        wandb.log({
            "baseline/loss":                  loss_no_smooth.item(),
            "baseline/entropy_no_smooth":     entr,
            "baseline/participation_no_smooth": particip,
        })

    w_std = W.std().item()
    d_r   = nn.Parameter(torch.randn(W.shape[0], device=W.device, dtype=W.dtype) * w_std * 0.1)
    d_c   = nn.Parameter(torch.randn(W.shape[1], device=W.device, dtype=W.dtype) * w_std * 0.1)

    return W, X, d_r, d_c


def _build_LR(W, rank, d_r, d_c, scaling):
    """Convert optimised scaling parameters into L, R matrices in original space."""
    D_r, D_c, D_r_inv, D_c_inv = None, None, None, None
    if scaling in ("rows", "both"):
        D_r     = torch.diag(torch.exp(d_r))
        D_r_inv = torch.diag(torch.exp(-d_r))
    if scaling in ("cols", "both"):
        D_c     = torch.diag(torch.exp(d_c))
        D_c_inv = torch.diag(torch.exp(-d_c))

    if scaling == "cols":
        W_fs = W @ D_c
    elif scaling == "rows":
        W_fs = D_r @ W
    else:
        W_fs = D_r @ W @ D_c

    U_f, S_f, V_f   = torch.linalg.svd(W_fs, full_matrices=False)
    U_t, S_t, V_t   = truncate_svd(U_f, S_f, V_f, rank)

    sqrt_S  = torch.sqrt(torch.diag(S_t))
    L_scaled = U_t @ sqrt_S
    R_scaled = sqrt_S @ V_t

    if scaling == "cols":
        L, R = L_scaled, R_scaled @ D_c_inv
    elif scaling == "rows":
        L, R = D_r_inv @ L_scaled, R_scaled
    else:
        L, R = D_r_inv @ L_scaled, R_scaled @ D_c_inv

    return L, R


def train_svd_scalers_simultaneously(W, X, rank, args, num_epochs=40, lr=1e-2, layername=None, device=None):
    """
    Optimize weight compression using learnable singular value scaling.
    Trains d_r and d_c **simultaneously** to minimise reconstruction error on X.

    Args:
        W:          Original weight matrix  (d_out x d_in)
        X:          Activation data         (batch x tokens x d_in)
        rank:       Target rank
        args:       Argument namespace
        num_epochs: Training epochs
        lr:         Learning rate (overridden by args.learning_rate)
        device:     Target device

    Returns:
        L, R:          Low-rank factors such that W ≈ L @ R
        loss_history:  Per-epoch loss values
    """
    print("Training SVD compressor with simultaneous optimization of d_r and d_c")
    if device is None:
        device = W.device

    W, X, d_r, d_c = _init_parameters_and_baseline(W, X, rank, args, device)
    scaling         = args.scaling_algo

    # Determine which parameters to optimise
    if scaling == "cols":
        params = [d_c]
    elif scaling == "rows":
        params = [d_r]
    else:  # both
        params = [d_r, d_c]

    loss_history = _run_svd_training_phase(
        W, X, rank, args,
        d_r=d_r, d_c=d_c,
        params_to_train=params,
        scaling=scaling,
        num_epochs=num_epochs,
        lr=lr,
        phase_name=layername,
    )

    with torch.no_grad():
        L, R         = _build_LR(W, rank, d_r, d_c, scaling)
        W_compressed = L @ R
        final_loss   = loss_fn_wprime(W, W_compressed, X, mode="frobenius")
        print(f"Final Loss after simultaneous training: {final_loss.item():.6f}")

    return L, R, loss_history


def train_svd_scalers_sequentially(W, X, rank, args, num_epochs=40, lr=1e-2, layername=None, device=None):
    """
    Alternate optimization of low-rank factors for weight compression.

    Trains d_c **fully** in Phase 1, then trains d_r **fully** in Phase 2
    (with d_c held fixed).  Only meaningful when args.scaling_algo == "both";
    falls back to simultaneous training otherwise.

    Args:
        W:          Original weight matrix  (d_out x d_in)
        X:          Activation data         (batch x tokens x d_in)
        rank:       Target rank
        args:       Argument namespace
        num_epochs: Epochs **per phase**
        lr:         Learning rate (overridden by args.learning_rate)
        device:     Target device

    Returns:
        L, R:          Low-rank factors such that W ≈ L @ R
        loss_history:  Combined per-epoch loss values (phase 1 then phase 2)
    """
    if device is None:
        device = W.device

    W, X, d_r, d_c = _init_parameters_and_baseline(W, X, rank, args, device)
    scaling         = args.scaling_algo

    if scaling != "both":
        # Nothing to alternate — delegate to the simultaneous trainer
        print(
            f"Warning: alternating optimisation requested but scaling='{scaling}'. "
            "Falling back to simultaneous training."
        )
        return train_svd_compressor(W, X, rank, args, num_epochs=num_epochs, lr=lr, device=device)

    # ------------------------------------------------------------------ #
    # Phase 1: optimise d_r only (d_c frozen at its initial value)        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 1 — optimising d_r (rows) with d_c frozen")
    print("=" * 60)
    loss_history_phase1 = _run_svd_training_phase(
        W, X, rank, args,
        d_r=d_r, d_c=d_c,
        params_to_train=[d_r],
        scaling="rows",          # only column scaling active this phase
        num_epochs=num_epochs,
        lr=lr,
        phase_name="phase1_rows",
    )

    # ------------------------------------------------------------------ #
    # Phase 2: optimise d_c only (d_r now fixed from Phase 1)             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PHASE 2 — optimising d_c (cols) with d_r fixed")
    print("=" * 60)
    loss_history_phase2 = _run_svd_training_phase(
        W, X, rank, args,
        d_r=d_r, d_c=d_c,
        params_to_train=[d_c],
        scaling="both",          # both scalings applied; only d_r receives gradients
        num_epochs=num_epochs,
        lr=lr,
        phase_name="phase2_cols",
    )

    with torch.no_grad():
        L, R         = _build_LR(W, rank, d_r, d_c, scaling="both")
        W_compressed = L @ R
        final_loss   = loss_fn_wprime(W, W_compressed, X, mode="frobenius")
        print(f"Final Loss after alternating training: {final_loss.item():.6f}")

    return L, R, loss_history_phase1 + loss_history_phase2