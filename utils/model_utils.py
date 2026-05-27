# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import os
import torch
import torch.nn as nn
import wandb
import time



def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def get_layer_io_save_path(args):
    return os.path.join(args.save_path, "layer_io", f"{args.layer_idx:03d}.pt")


def capture_layer_io(layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())

        return hook

    handles = []

    captured_inputs = {
        "k_proj": [],  # q_proj, v_proj has the same input as k_proj
        "o_proj": [],
        "gate_proj": [],  # up_proj has the same input as gate_proj
        "down_proj": [],
    }

    captured_outputs = {
        "v_proj": [],
    }

    for name in captured_inputs.keys():
        module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
        handles.append(
            module.register_forward_hook(hook_factory(name, captured_inputs, True))
        )

    for name in captured_outputs.keys():
        module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
        handles.append(
            module.register_forward_hook(hook_factory(name, captured_outputs, False))
        )

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx : seq_idx + 1].to("cuda")
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {"input": captured_inputs, "output": captured_outputs}


def save_model(model, model_name, checkpoint=None, **kwargs):
    """
    Save the model to disk in a format that can be reloaded with
    ``SVDLlamaForCausalLM.from_pretrained()`` or
    ``AutoModelForCausalLM.from_pretrained()``.

    How it works
    ------------
    If the model contains SVDLinear layers, an ``SVDLlamaConfig`` is built
    from the model's existing config and extended with ``svd_layers_config``
    (a ``{module_path: rank}`` dict for every compressed layer).

    The config is saved explicitly with ``config.save_pretrained()``, and the
    raw PyTorch state dict is written with ``safetensors`` (preferred) or
    ``torch.save``.  This avoids the class/config mismatch that would occur
    from calling ``model.save_pretrained()`` on a ``LlamaForCausalLM`` instance
    that has a ``SVDLlamaConfig`` temporarily injected into it.

    When loading, ``SVDLlamaForCausalLM.__init__`` reads ``svd_layers_config``
    and reconstructs the same SVDLinear topology before state-dict loading, so
    HuggingFace finds matching keys everywhere.
    """
    from modules.linears import SVDLinear, SVDLlamaConfig, SVDQwenConfig

    wandb_run_id = None if wandb.run is None else wandb.run.id
    wandb_sweep_id = None if wandb.run is None else wandb.run.sweep_id
    sweep_config = None
    if wandb_sweep_id is not None:
        sweep_api = wandb.Api().sweep(f"{wandb.run.entity}/{wandb.run.project}/{wandb_sweep_id}")
        if sweep_api is not None:
            sweep_config = sweep_api.config

    output_dir = _resolve_model_dir(model_name, checkpoint, wandb_run_id, wandb_sweep_id)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build config
    # ------------------------------------------------------------------
    svd_layers_config = {
        path: module.truncation_rank
        for path, module in model.named_modules()
        if isinstance(module, SVDLinear)
    }

    if svd_layers_config:
        print(f"Saving {len(svd_layers_config)} SVD-compressed layer(s) "
              f"with SVDConfig.")
        config_dict = model.config.to_dict()
        config_dict.pop("model_type", None)
        config_dict["svd_layers_config"] = svd_layers_config
        
        if model.config.model_type == "qwen3":
            print("Detected Qwen model, using SVDQwenConfig.")
            save_config = SVDQwenConfig(**config_dict)
        elif model.config.model_type == "llama":
            print("Detected LLaMA model, using SVDLlamaConfig.")
            save_config = SVDLlamaConfig(**config_dict)
    else:
        save_config = model.config

    # Save config directly — no model instance involved, no class mismatch.
    save_config.save_pretrained(output_dir)

    # ------------------------------------------------------------------
    # Save raw state dict
    # Prefer safetensors (faster mmap loading, no arbitrary code execution).
    # Fall back to torch.save if the library is not installed.
    # ------------------------------------------------------------------
    import json as _json

    _MAX_SHARD_BYTES = 5 * 1024 ** 3  # 5 GB per shard

    state_dict = model.state_dict()
    try:
        from safetensors.torch import save_file as _st_save

        # Split state dict into shards
        shards, current, cur_bytes = [], {}, 0
        for key, tensor in state_dict.items():
            nb = tensor.numel() * tensor.element_size()
            if current and cur_bytes + nb > _MAX_SHARD_BYTES:
                shards.append(current)
                current, cur_bytes = {}, 0
            current[key] = tensor
            cur_bytes += nb
        if current:
            shards.append(current)

        if len(shards) == 1:
            weights_path = os.path.join(output_dir, "model.safetensors")
            _st_save(shards[0], weights_path)
            print(f"Saved {len(state_dict)} tensors → {weights_path}")
        else:
            n = len(shards)
            total_size, weight_map = 0, {}
            for i, shard in enumerate(shards, 1):
                fname = f"model-{i:05d}-of-{n:05d}.safetensors"
                shard_path = os.path.join(output_dir, fname)
                _st_save(shard, shard_path)
                for k, t in shard.items():
                    weight_map[k] = fname
                    total_size += t.numel() * t.element_size()
            index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
            index_path = os.path.join(output_dir, "model.safetensors.index.json")
            with open(index_path, "w") as f:
                _json.dump(index, f, indent=2)
            print(f"Saved {len(state_dict)} tensors in {n} shards → {output_dir}")
            print(f"Shard index written to {index_path}")

    except ImportError:
        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(state_dict, weights_path)
        print(f"Saved {len(state_dict)} tensors → {weights_path}")

    print(f"Model saved to {output_dir}")

    # ------------------------------------------------------------------
    # Persist run / training configuration alongside the weights.
    # ------------------------------------------------------------------
    model_args = kwargs.get("model_args", None)
    training_args = kwargs.get("training_args", None)
    ptq_args = kwargs.get("ptq_args", None)

    if model_args is not None:
        with open(os.path.join(output_dir, "model_args.txt"), "w") as f:
            for key, value in model_args.__dict__.items():
                f.write(f"{key}: {value}\n")

    if training_args is not None:
        with open(os.path.join(output_dir, "training_args.txt"), "w") as f:
            for key, value in training_args.__dict__.items():
                f.write(f"{key}: {value}\n")

    if ptq_args is not None:
        with open(os.path.join(output_dir, "ptq_args.txt"), "w") as f:
            for key, value in ptq_args.__dict__.items():
                f.write(f"{key}: {value}\n")

    if sweep_config is not None:
        import yaml
        with open(os.path.join(output_dir, "sweep_config.yaml"), "w") as f:
            yaml.dump(sweep_config, f)


def _resolve_model_dir(model_name, checkpoint=None, wandb_run_id=None, wandb_sweep_id=None):
    """Reconstruct the output directory that save_model wrote to."""
    output_dir = os.path.dirname(__file__) + "/../output_dir/saved_models/"
    model_dir = model_name.replace("/", "_")

    if wandb_sweep_id is not None:
        output_dir = os.path.join(output_dir, f"sweeps/sweep_{wandb_sweep_id}/")
    if wandb_run_id is not None:
        model_dir += f"_{wandb_run_id}"
    if checkpoint is not None:
        model_dir += f"_chkpt-{checkpoint}"

    return os.path.abspath(os.path.join(output_dir, model_dir))


def load_model(
    model_name,
    checkpoint=None,
    wandb_run_id=None,
    wandb_sweep_id=None,
    model_path=None,
    device_map="auto",
):
    """
    Load a model that was previously saved with ``save_model()``.

    Parameters
    ----------
    model_name : str
        Same value passed to ``save_model()``, used to reconstruct the
        output directory (e.g. ``"meta-llama/Llama-3.1-8B"``).
    checkpoint : int or str, optional
        Checkpoint suffix used at save time.
    wandb_run_id : str, optional
        WandB run ID used at save time (needed to match the directory name).
    wandb_sweep_id : str, optional
        WandB sweep ID used at save time.
    model_path : str, optional
        Absolute path to the saved directory.  When provided, all of the
        above path-construction arguments are ignored.
    device_map : str or dict
        Passed directly to ``from_pretrained``.  ``"auto"`` spreads the
        model across available GPUs/CPU as needed.
    torch_dtype : torch.dtype
        Dtype to load weights in.  Defaults to ``bfloat16``.

    Returns
    -------
    model
        Either a ``SVDLlamaForCausalLM`` (if the checkpoint was saved with
        SVD layers) or a plain ``LlamaForCausalLM`` / ``AutoModelForCausalLM``
        instance, ready for inference or further training.
    """
    # Importing SVDLlamaForCausalLM registers SVDLlamaConfig and
    # SVDLlamaForCausalLM with HuggingFace's Auto classes as a side-effect,
    # so AutoModelForCausalLM.from_pretrained can dispatch correctly.
    from modules.linears import SVDLlamaForCausalLM  # noqa: F401
    from transformers import AutoModelForCausalLM

    load_dir = model_path or _resolve_model_dir(
        model_name, checkpoint, wandb_run_id, wandb_sweep_id
    )

    if not os.path.isdir(load_dir):
        raise FileNotFoundError(
            f"Model directory not found: {load_dir}\n"
            "Make sure model_name / checkpoint / wandb IDs match those used at save time, "
            "or pass model_path directly."
        )

    # Peek at config.json to report what we're loading.
    config_path = os.path.join(load_dir, "config.json")
    model_type = "unknown"
    if os.path.isfile(config_path):
        import json
        with open(config_path) as f:
            model_type = json.load(f).get("model_type", "unknown")

    print(f"Loading {model_type!r} model from {load_dir} ...")

    model = AutoModelForCausalLM.from_pretrained(
        load_dir,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    n_svd = sum(1 for _, m in model.named_modules() if m.__class__.__name__ == "SVDLinear")
    if n_svd:
        print(f"Loaded model with {n_svd} SVDLinear layer(s).")
    else:
        print("Loaded standard (non-SVD) model.")

    return model