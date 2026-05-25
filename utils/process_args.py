# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
import json
from typing import Optional, Tuple

import argparse
import transformers
import wandb

@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="test-input", metadata={"help": "Input model"}
    )
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"}
    )
    optimized_rotation_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"}
    )
    access_token: Optional[str] = field(
        default="49c8e7dca82f91f9d65021c3dd71101b686c1f53",
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )
    
def nsamples_type(x):
    if x.lower() == "full":
        return "full"
    if x.isdigit():
        return int(x)
    raise argparse.ArgumentTypeError(
        "Must be a positive integer or 'full'"
    )


def parser_gen():
    """ Implement all args here which are NOT 1:1 part of ModelArguments or TrainingArguments """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    # --- Hadamard Control Group ---
    # Note: These control the ONLINE R3/R4 hadamard parts.
    # --rotate flag controls the base offline R1/R2 rotations.

    parser.add_argument(
        "--run_analysis",
        action="store_true",
        help="Run activations analysis on the weights quantized model."
    )

    parser.add_argument(
        "--quantize_only_module",
        type=str,
        default=None,
        help="The full name of a single module to quantize (e.g., 'model.layers.15.mlp.down_proj'). All others will remain in full precision."
    )

    parser.add_argument(
        "--hadamard_online",
        action="store_true",
        help="Enable GLOBAL online Hadamard (R4 for down_proj + R3 if k_bits<16)."
    )
    parser.add_argument(
        "--selective_had_layers_path",
        type=str,
        default=None,#"llama2_7b_rotate_layers_p5.json",
        help="Enable SELECTIVE online Hadamard (R4 for down_proj listed in JSON + R3 if k_bits<16)."
               " Path to JSON file containing 'layers_to_rotate' list."
    )
    parser.add_argument(
        "--online_r3_only",
        action="store_true",
        help="Enable ONLY R3 online Hadamard (if k_bits<16), disable R4 online Hadamard."
    )
    # --- End Hadamard Control Group ---
    
    '''parser.add_argument(
        "--hadamard_online",
        action="store_true",
        help="Enable online Hadamard rotations (R3, R4) for SpinQuant_had configuration."
    )
    parser.add_argument(
        "--selective_had_layers_path",
        type=str,
        default=None,
        help="Path to JSON file containing list of layer indices for selective online Hadamard rotation."
    )'''
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.95,
        help="The cumulative probability threshold for discovering the optimal k."
    )
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="The block size to use for block-wise mixed-precision quantization (e.g., 128, 64, 32)."
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    parser.add_argument(
    "--top_k",
    type=int,
    default=20,
    help="The number of top-k tokens to consider for the JSD/Jaccard analysis."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The weighting factor for the hybrid score (weight for confidence shift)."
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--load_qmodel_path",
        type=str,
        default=None,
        help="Load the quantized model from the specified path!",
    )
    parser.add_argument(
        "--save_qmodel_path",
        type=str,
        default=None,
        help="Save the quantized model to the specified path!",
    )
    parser.add_argument(
        "--export_to_et",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the quantized model to executorch and save in save_qmodel_path",
    )

    # Experiments Arguments
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )
    parser.add_argument(
        "--nb_eval_runs",
        type=int,
        default=1,
        help="Number of evaluation runs",
    )
    parser.add_argument(
        "--timing_output_path",
        type=str,
        default=None,
        help="Path to save the detailed timing and PPL results in JSON format."
    )
    parser.add_argument(
        "--exclude_activations_layers", # Renamed for clarity
        type=int,
        nargs="+",
        default=None,
        help="A list of layer indices to EXCLUDE from activation quantization. If not provided, all activations are quantized."
    )
    parser.add_argument(
        "--high_precision_bits",
        type=int,
        default=16,
        help="The bit-width to use for layers specified as high precision."
    )
    parser.add_argument(
        "--mixed_precision_config",
        type=str,
        default=None,
        help="Path to a JSON file specifying which layer IDs should be in high precision."
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="wikitext2",
        help="The dataset to use for evaluation. Options: 'wikitext2', 'c4'."
    )
    parser.add_argument(
        "--noise_scalar",
        type=float,
        default=0,
        help="Noise scaling factor",
    )
    parser.add_argument(
        "--wandb_sweep",
        type=str,
        help="Specify a sweep config file to run this as a Weights and Biases sweep",
        default=None,
    )
    parser.add_argument(
        "--wandb_run",
        action="store_true",
        help="Run this as a Weights and Bias run",
        default=False,
    )
    parser.add_argument(
        "--use_sensitivity_cache",
        type=str,
        default=None,
        help="Path to sensitivity cache file",
    )
    parser.add_argument(
        "--test_loader_nsamples",
        type=nsamples_type,
        default=4096,
        help="Number of samples for the test data loader",
    )
    parser.add_argument(
        "--test_loader_seqlen",
        type=int,
        default=256,
        help="Sequence length for the test data loader",
    )
    parser.add_argument(
        "--train_loader_nsamples",
        type=nsamples_type,
        default=4096,
        help="Number of samples for the training data loader",
    )
    parser.add_argument(
        "--train_loader_seqlen",
        type=int,
        default=256,
        help="Sequence length for the training data loader",
    )
    parser.add_argument(
        "--train_bs",
        type=int,
        default=4,
        help="Batch size for trainer data loader",
    )
    parser.add_argument(
        "--eval_bs",
        type=int,
        default=4,
        help="Batch size for trainer data loader",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=1.0,
        help="Target parameter count ratio for mixed-precision quantization, 0-1",
    )
    parser.add_argument(
        "--use_alternating_LR_training",
        action="store_true",
        help="Use alternating low-rank component training during knowledge distillation",
        default=False,
    )
    parser.add_argument(
        "--apply_svd_smoothing",
        action="store_true",
        help="Bypass creation of SVD layers, leave to false if just fine-tuning, e.g. pretrained SVD layers",
        default=False,
    )
    parser.add_argument(
        "--train_low_rank_smoothing",
        action="store_true",
        help="Train a smoothing function for better low-rank decomposition",
        default=False,
    )
    parser.add_argument(
        "--train_per_layer",
        action="store_true",
        help="Train the model gradually, layer-by-layer",
        default=False,
    )
    parser.add_argument(
        "--use_distillation",
        action="store_true",
        help="Use knowledge distillation during quantization",
        default=False,
    )
    parser.add_argument(
        "--fine_tune_after_compression",
        action="store_true",
        help="Fine-tune the model after compression",
        default=False,
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.005,
        help="Learning rate for training the smoothing function or low-rank components",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
        help="Number of epochs for training the smoothing function or low-rank components",
    )
    parser.add_argument(
        "--scaling_algo",
        type=str,
        default='both',
        help="rows, cols, or both for the scaling in the SVD smoother training",
    )
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="adamw",
        help="Optimizer to use for training the smoothing function or low-rank components (e.g., 'adam', 'sgd')",
    )
    parser.add_argument(
        "--scheduler_name",
        type=str,
        default=None,
        help="Scheduler to use for training the smoothing function or low-rank components (e.g., 'cosine', 'step', 'warmup_cosine')",
    )
    parser.add_argument(
        "--l2_regularizer_scale",
        type=float,
        default=0.000001,
        help="L2 regularization scale to apply to the low-rank components during training of the smoothing function or low-rank components",
    )
    parser.add_argument(
        "--add_regularizing_noise",
        type=float,
        default=1e-6,
        help="Regularizing noise to stabilize SVD during training",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type (e.g., 'linear', 'cosine') for training the smoothing function or low-rank components",
    )
    parser.add_argument(
        "--add_term_to_loss",
        type=str,
        default=None,
        help="Additional term to add to the loss during training of the smoothing function, 'entropy' for effective rank entropy minimization, 'participation' for effective rank participation ratio minimization",
    )
    parser.add_argument(
        "--term_loss_scaler",
        type=float,
        default=None,
        help="Scaling factor to apply to the additional loss term (e.g., effective rank entropy or participation ratio) during training of the smoothing function",
    )
    parser.add_argument(
        "--intercept_sanitize_grad",
        type=bool,
        default=False,
        help="Whether to sanitize gradients for the intercept term in the SVD smoother training, which can help stabilize training"
    )
    
    parser.add_argument(
        "--compress_specific_module",
        type=str,
        default=None,
        help="The full name of a single module to apply compression to in format model.layers.31.self_attn.q_proj. If not provided, compression is applied to all modules.",
    )
    
    parser.add_argument(
        "--optimizer_constraint",
        type=str,
        default="l2",
        help="Constraint to apply to the optimizer updates during training of the smoothing function or low-rank components, 'clamp', 'l2', or None ",
    )
    
    parser.add_argument(
        "--compress_specific_layers",
        type=json.loads,
        default=None,
        help="Define a list of layers to apply compression to in format [0, 1, 2, 3, 4] or None to apply to all layers",
    )
    
    parser.add_argument(
        "--compressed_model",
        type=str,
        default=None,
        help="Name or path of the model which will be compressed",
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        default="wikitext2",
        help="Dataset used for fine-tuning, options: alpaca or wikitext2",
    )
    
    parser.add_argument(
        "--kd_epochs",
        type=int,
        default=1,
        help="Num epochs for KD fine tuning",
    )
    
    parser.add_argument(
        "--kd_alpha",
        type=float,
        default=1,
        help="Balance between KD and task loss",
    )
    
    parser.add_argument(
        "--num_paraphrases_trainset",
        type=int,
        default=3,
        help="Number of paraphrases to use for a training set if applicable (Alpaca)",
    )
    
    parser.add_argument(
        "--train_svd_scalers_sequentially",
        type=bool,
        default=False,
        help="Trains SVD scalers sequentially, one at a time, instead of all at once.",
    )
    parser.add_argument(
        "--save_svd_model",
        type=bool,
        default=False,
        help="Save the SVD model (U, S, V) after training the scalers",
    )
    
    parser.add_argument(
        "--run_all_evals",
        type=bool,
        default=False,
        help="Run full evals",
    )
    
    parser.add_argument(
        "--num_evals_per_epoch_zeroshot",
        type=int,
        default=0,
        help="Number of evaluation runs per epoch during zero-shot evaluation with knowledge distillation (set to 0 to disable intermediate evals and only run final eval at the end of training).",
    )
    
    

    args, unknown = parser.parse_known_args()

    # --- Add validation for mutually exclusive logic (optional but good) ---
    # Although shell script enforces passing only one, Python can double-check.
    had_flags_count = sum([
        args.hadamard_online,
        args.selective_had_layers_path is not None,
        args.online_r3_only
    ])
    if had_flags_count > 1:
        parser.error("Only one of --hadamard_online, --selective_had_layers_path, or --online_r3_only can be specified.")

    # assert (
    #     args.a_groupsize == args.w_groupsize
    # ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope is False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown

def parse_wandb_sweep(ptq_args, unknown_args: dict):
    if ptq_args.wandb_sweep is False:
        return ptq_args, unknown_args

    # Overwrite where possible, what's left add to ptq_args
    wandb_args = wandb.config.as_dict()
    overwritten_keys = []
    for key, value in wandb_args.items():
        if hasattr(ptq_args, key):
            setattr(ptq_args, key, wandb_args[key])
            print(f"Updated {key} from wandb config: {value}")
            overwritten_keys.append(key)
            continue
        
        # hacky for unknown args...
        if "--" + key in unknown_args:
            i = unknown_args.index("--" + key) + 1
            unknown_args[i] = str(wandb_args[key])
            print(f"Updated {key} in unknown args from wandb config: {value}")
            overwritten_keys.append(key)
            continue
    
    # remove overwritten keys, add the rest to ptq_args
    [wandb_args.pop(k) for k in overwritten_keys]
    [setattr(ptq_args, k, v) for k, v in wandb_args.items()]
    
    return ptq_args, unknown_args


def process_args_ptq():
    ptq_args = None 

    ptq_args, unknown_args = parser_gen()
    

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    print(f"Optimized rotation path: {model_args.optimized_rotation_path}")
    print(f"Selective hadamard layers path: {ptq_args.selective_had_layers_path}")
    if model_args.optimized_rotation_path is not None:
        ptq_args.optimized_rotation_path = model_args.optimized_rotation_path
    else:
        ptq_args.optimized_rotation_path = None
    ptq_args.bsz = training_args.per_device_eval_batch_size
    ptq_args.input_model = model_args.input_model
    
    if wandb.run is not None:
        ptq_args, unknown_args = parse_wandb_sweep(ptq_args, unknown_args)
    
    # parse config if WandB sweep

    return model_args, training_args, ptq_args