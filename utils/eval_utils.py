# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from copy import deepcopy
import logging
import os
import json
import re

import torch
import torch.cuda
import time
from tqdm import tqdm
import argparse
import numpy as np
import math
import random
from pathlib import Path
import gc
import json
from datetime import datetime
import logging
import lm_eval
from modules.lm_eval_wrappers import MyCustomLM
import datasets
import sys

from utils import model_utils

def calculate_per_layer_bops(model, args):
    """
    Calculates the total computational cost for each of the 32 layers in the model,
    aggregating the costs of all linear sub-modules within each layer, based on its quantization
    configuration. The cost is modeled as Bit-Operations (BOPs), where for each
    linear layer, BOPs = base_macs * weight_bits * activation_bits.

    Returns:
        dict: A dictionary mapping layer index (0-31) to its total BOPs value.
    """
    # Initialize a dictionary to hold BOPs for each layer
    num_layers = model.config.num_hidden_layers
    layer_bops = {i: 0 for i in range(num_layers)}
    
    w_bits = args.w_bits if args.w_bits < 16 else 16

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Find which of the 32 layers this linear module belongs to
            match = re.search(r'model\.layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))

                # --- Calculate BOPs for this specific nn.Linear module ---
                base_macs = module.in_features * module.out_features
                
                # Determine activation bits for this layer's input
                a_bits = args.a_bits if args.a_bits < 16 else 16
                if (
                    hasattr(args, 'exclude_activations_layers') and
                    args.exclude_activations_layers is not None and
                    layer_idx in args.exclude_activations_layers
                ):
                    a_bits = 16
                elif "down_proj" in name and hasattr(args, 'int8_down_proj') and args.int8_down_proj:
                    a_bits = 8
                
                module_bops = base_macs * w_bits * a_bits
                
                # Accumulate the BOPs for the corresponding main layer
                if layer_idx in layer_bops:
                    layer_bops[layer_idx] += module_bops
    
    return layer_bops

@torch.no_grad()
def get_logits_for_analysis(model, testloader, dev, args):
    """
    Performs an efficient, layer-by-layer forward pass on a set of samples
    to get the final logits. Mimics the core logic of the main evaluator.
    """
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    seq_len = model.seqlen
    input_ids = testloader.input_ids
    nsamples = input_ids.numel() // seq_len
    input_ids = input_ids[:, : nsamples * seq_len].view(nsamples, seq_len).to(dev)

    batch_size = args.bsz
    nbatches = (nsamples + batch_size - 1) // batch_size
    
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module): super().__init__(); self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(0, nsamples, batch_size):
        batch = input_ids[i:i+batch_size]
        try: model(batch)
        except ValueError: pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in tqdm(range(len(layers)), desc=" (Forward Pass) Layers"):
        layer = layers[i].to(dev)
        for j in range(nbatches):
            outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    all_logits = []
    for i in range(nbatches):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        all_logits.append(logits.cpu()) # Move to CPU to save VRAM

    model.config.use_cache = use_cache
    return all_logits

@torch.no_grad()
def _evaluator_multiple_gpus(model, testenc, dev, args):
    model.eval()
    dev = model.device

    print("INFO: nb of evaluation runs: ", args.nb_eval_runs)
    max_trials = args.nb_eval_runs

    list_total_inference_time = []
    list_time_per_token = []
    list_ppl = []

    list_total_inference_time_perf_count = []
    list_time_per_token_perf_count = []



    for _ in range(max_trials):

        use_cache = model.config.use_cache
        model.config.use_cache = False

        layers = model.model.layers
        seq_len = model.seqlen

        # Convert the whole text of evaluation dataset into batches of sequences.
        input_ids = testenc.input_ids  # (1, text_len)
        nsamples = input_ids.numel() // seq_len  # The tail is truncated.
        input_ids = (
            input_ids[:, : nsamples * seq_len].view(nsamples, seq_len)
        )  # (nsamples, seqlen)
        
        total_tokens_processed = nsamples * seq_len

        print(f"INFO: Evaluator using seqlen={seq_len}, found {nsamples} samples ({total_tokens_processed} tokens).")

        batch_size = args.bsz
        input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
        nbatches = len(input_ids)

        dtype = next(iter(model.parameters())).dtype
        # The input of the first decoder layer.
        inps = [None] * nbatches
        cache = {"i": 0, "attention_mask": None}

        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher(layers[0])

        for i in range(nbatches):
            batch = input_ids[i]
            try:
                model(batch.to(model.device))
            except ValueError:
                pass
        layers[0] = layers[0].module

        position_ids = cache["position_ids"]
        attention_mask = cache["attention_mask"]
        
        cache.clear()
        outs = [0] * nbatches
        

        # --- Timing Initialization ---
        total_layer_processing_time_ms = 0.0
        total_lm_head_time_ms = 0.0
        start_event_layer_loop = torch.cuda.Event(enable_timing=True)
        end_event_layer_loop = torch.cuda.Event(enable_timing=True)
        start_event_head_loop = torch.cuda.Event(enable_timing=True)
        end_event_head_loop = torch.cuda.Event(enable_timing=True)
        # --- End Timing Initialization ---

        torch.cuda.synchronize()
        start_event_layer_loop.record()
        start_cpu_layer = time.perf_counter()

        for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
            layer = layers[i]

            # Dump the layer input and output
            if args.capture_layer_io and args.layer_idx == i:
                captured_io = model_utils.capture_layer_io(layer, inps)
                save_path = model_utils.get_layer_io_save_path(args)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(captured_io, save_path)
                logging.info(f"Dumped layer input and output to: {save_path}")

            for j in range(nbatches):
                tmp = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    #  defined.
                    position_ids=position_ids,
                )[0]
                outs[j] = tmp
            
            del layer
            inps, outs = outs, inps
            
        end_event_layer_loop.record()
        torch.cuda.synchronize()
        end_cpu_layer = time.perf_counter()
        total_layer_processing_time_ms_cpu = (end_cpu_layer - start_cpu_layer) * 1000
        print(f"INFO: Layer processing loop finished. Time (CPU): {total_layer_processing_time_ms_cpu:.2f} ms")
        total_layer_processing_time_ms = start_event_layer_loop.elapsed_time(end_event_layer_loop)
        print(f"INFO: Layer processing loop finished. Time: {total_layer_processing_time_ms:.2f} ms")
        # --- End Layer Processing Loop ---


        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        print("INFO: Starting LM Head and Loss calculation loop...")
        torch.cuda.synchronize()
        start_event_head_loop.record()
        start_cpu_head = time.perf_counter()

        for i in range(nbatches):
            hidden_states = inps[i]
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = input_ids[i][:, 1:].to(shift_logits.device)
            
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1).detach().cpu()
            nlls.append(neg_log_likelihood)
            
        
        del inps, outs
        torch.cuda.empty_cache()
        end_event_head_loop.record()
        torch.cuda.synchronize()
        end_cpu_head = time.perf_counter()
        total_lm_head_processing_time_cpu = (end_cpu_head - start_cpu_head) * 1000
        print(f"INFO: LM Head and Loss calculation loop finished. Time (CPU): {total_lm_head_processing_time_cpu:.2f} ms")
        total_lm_head_time_ms = start_event_head_loop.elapsed_time(end_event_head_loop)
        print(f"INFO: LM Head loop finished. Time: {total_lm_head_time_ms:.2f} ms")
        # --- End LM Head Loop ---

        nlls_tensor = torch.cat(nlls)
        ppl = torch.exp(nlls_tensor.mean())
        model.config.use_cache = use_cache
        logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")

        # --- Timing Report ---
        total_inference_time_ms = total_layer_processing_time_ms + total_lm_head_time_ms
        total_inference_time_ms_cpu = total_layer_processing_time_ms_cpu + total_lm_head_processing_time_cpu
        if total_tokens_processed > 0:
            time_per_token_ms = total_inference_time_ms / total_tokens_processed
            time_per_token_ms_cpu = total_inference_time_ms_cpu / total_tokens_processed
            print(f"Total Inference Time (Layers + LM Head) (CPU): {total_inference_time_ms_cpu:.2f} ms")
            print(f"Average Inference Time per Token (CPU): {time_per_token_ms_cpu:.4f} ms/token")
            print(f"Total Inference Time (Layers + LM Head): {total_inference_time_ms:.2f} ms")
            print(f"Average Inference Time per Token: {time_per_token_ms:.4f} ms/token")
            list_total_inference_time.append(total_inference_time_ms)
            list_total_inference_time_perf_count.append(total_inference_time_ms_cpu)
            list_time_per_token.append(time_per_token_ms)
            list_time_per_token_perf_count.append(time_per_token_ms_cpu)
            list_ppl.append(ppl.item())
        else:
            print("No tokens processed, cannot calculate time per token.")
        # --- End Timing Report ---


    avg_total_inference_time = sum(list_total_inference_time) / len(list_total_inference_time)
    var_total_inference_time = sum(
        [(x - avg_total_inference_time) ** 2 for x in list_total_inference_time]
    ) / len(list_total_inference_time)
    avg_total_inference_time_perf_count = sum(list_total_inference_time_perf_count) / len(list_total_inference_time_perf_count)
    var_total_inference_time_perf_count = sum(
        [(x - avg_total_inference_time_perf_count) ** 2 for x in list_total_inference_time_perf_count]
    ) / len(list_total_inference_time_perf_count)

    avg_time_per_token = sum(list_time_per_token) / len(list_time_per_token)
    var_time_per_token = sum(
        [(x - avg_time_per_token) ** 2 for x in list_time_per_token]
    ) / len(list_time_per_token)
    avg_time_per_token_perf_count = sum(list_time_per_token_perf_count) / len(list_time_per_token_perf_count)
    var_time_per_token_perf_count = sum(
        [(x - avg_time_per_token_perf_count) ** 2 for x in list_time_per_token_perf_count]
    ) / len(list_time_per_token_perf_count)

    avg_ppl = sum(list_ppl) / len(list_ppl)

    # --- Save Detailed Results to JSON if path is provided ---
    if hasattr(args, 'timing_output_path') and args.timing_output_path:
        print(f"INFO: Saving detailed timing results to {args.timing_output_path}")
        results_to_save = {
            "nb_runs": max_trials,
            "list_ppl": list_ppl,
            "avg_ppl": avg_ppl,
            "cuda_timing_ms": {
                "list_total_time": list_total_inference_time,
                "list_token_time": list_time_per_token,
                "avg_total_time": avg_total_inference_time,
                "var_total_time": var_total_inference_time,
                "avg_token_time": avg_time_per_token,
                "var_token_time": var_time_per_token,
            },
            "cpu_timing_ms": {
                "avg_total_time": avg_total_inference_time_perf_count,
                "var_total_time": var_total_inference_time_perf_count,
                "avg_token_time": avg_time_per_token_perf_count,
                "var_token_time": var_time_per_token_perf_count,
            }
        }
        try:
            with open(args.timing_output_path, 'w') as f:
                json.dump(results_to_save, f, indent=4, default=lambda o: '<not serializable>')
            print(f"INFO: Successfully saved timing results to {args.timing_output_path}")
        except Exception as e:
            print(f"ERROR: Could not save timing results to {args.timing_output_path}: {e}")
    # --- End JSON Saving ---

    print(f"Average Total Inference Time: {avg_total_inference_time:.2f} ms")
    print(f"Average Total Inference Time (CPU): {avg_total_inference_time_perf_count:.2f} ms")
    print(f"Variance Total Inference Time: {var_total_inference_time:.2f} ms")
    print()
    print(f"Average Time per Token: {avg_time_per_token:.4f} ms/token")
    print(f"Average Time per Token (CPU): {avg_time_per_token_perf_count:.4f} ms/token")
    print(f"Variance Time per Token: {var_time_per_token:.4f} ms/token")
    print()
    print(f"Average PPL: {avg_ppl:.3f}")
    return avg_ppl, avg_time_per_token

def evaluator(model, testenc, dev, args):
    if torch.cuda.device_count() > 1:
        return _evaluator_multiple_gpus(model, testenc, dev, args)
    else:
        return _evaluator_single_gpu(model, testenc, dev, args)

@torch.no_grad()
def _evaluator_single_gpu(model, testenc, dev, args):
    model.eval()
    dev = model.device

    print("INFO: nb of evaluation runs: ", args.nb_eval_runs)
    max_trials = args.nb_eval_runs

    list_total_inference_time = []
    list_time_per_token = []
    list_ppl = []

    list_total_inference_time_perf_count = []
    list_time_per_token_perf_count = []



    for _ in range(max_trials):

        use_cache = model.config.use_cache
        model.config.use_cache = False

        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)

        layers[0] = layers[0].to(dev)

        seq_len = model.seqlen

        # Convert the whole text of evaluation dataset into batches of sequences.
        input_ids = testenc.input_ids  # (1, text_len)
        nsamples = input_ids.numel() // seq_len  # The tail is truncated.
        input_ids = (
            input_ids[:, : nsamples * seq_len].view(nsamples, seq_len).to(dev)
        )  # (nsamples, seqlen)
        
        total_tokens_processed = nsamples * seq_len

        print(f"INFO: Evaluator using seqlen={seq_len}, found {nsamples} samples ({total_tokens_processed} tokens).")

        batch_size = args.bsz
        input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
        nbatches = len(input_ids)

        dtype = next(iter(model.parameters())).dtype
        # The input of the first decoder layer.
        inps = [None] * nbatches
        cache = {"i": 0, "attention_mask": None}

        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

        layers[0] = Catcher(layers[0])

        for i in range(nbatches):
            batch = input_ids[i]
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache["position_ids"]
        attention_mask = cache["attention_mask"]
        
        cache.clear()
        torch.cuda.empty_cache()
        outs = [0] * nbatches
        

        # --- Timing Initialization ---
        total_layer_processing_time_ms = 0.0
        total_lm_head_time_ms = 0.0
        start_event_layer_loop = torch.cuda.Event(enable_timing=True)
        end_event_layer_loop = torch.cuda.Event(enable_timing=True)
        start_event_head_loop = torch.cuda.Event(enable_timing=True)
        end_event_head_loop = torch.cuda.Event(enable_timing=True)
        # --- End Timing Initialization ---

        torch.cuda.synchronize()
        start_event_layer_loop.record()
        start_cpu_layer = time.perf_counter()

        for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
            layer = layers[i].to(dev)

            # Dump the layer input and output
            if args.capture_layer_io and args.layer_idx == i:
                captured_io = model_utils.capture_layer_io(layer, inps)
                save_path = model_utils.get_layer_io_save_path(args)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(captured_io, save_path)
                logging.info(f"Dumped layer input and output to: {save_path}")

            for j in range(nbatches):
                # ensure all same device and dtype
                idev = inps[j].device
                pdev = position_ids.device
                
                inps[j] = inps[j].to(dev)
                position_ids = position_ids.to(dev)
                
                tmp = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    #  defined.
                    position_ids=position_ids,
                )[0]
                outs[j] = tmp
            layers[i] = layer.cpu()
            
            del layer
            inps, outs = outs, inps
            
        end_event_layer_loop.record()
        torch.cuda.synchronize()
        end_cpu_layer = time.perf_counter()
        total_layer_processing_time_ms_cpu = (end_cpu_layer - start_cpu_layer) * 1000
        print(f"INFO: Layer processing loop finished. Time (CPU): {total_layer_processing_time_ms_cpu:.2f} ms")
        total_layer_processing_time_ms = start_event_layer_loop.elapsed_time(end_event_layer_loop)
        print(f"INFO: Layer processing loop finished. Time: {total_layer_processing_time_ms:.2f} ms")
        # --- End Layer Processing Loop ---

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)

        model.lm_head = model.lm_head.to(dev)
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        print("INFO: Starting LM Head and Loss calculation loop...")
        torch.cuda.synchronize()
        start_event_head_loop.record()
        start_cpu_head = time.perf_counter()

        for i in range(nbatches):
            hidden_states = inps[i]
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = input_ids[i][:, 1:]
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1).detach().cpu()
            nlls.append(neg_log_likelihood)
            
        
        del inps, outs
        torch.cuda.empty_cache()
        end_event_head_loop.record()
        torch.cuda.synchronize()
        end_cpu_head = time.perf_counter()
        total_lm_head_processing_time_cpu = (end_cpu_head - start_cpu_head) * 1000
        print(f"INFO: LM Head and Loss calculation loop finished. Time (CPU): {total_lm_head_processing_time_cpu:.2f} ms")
        total_lm_head_time_ms = start_event_head_loop.elapsed_time(end_event_head_loop)
        print(f"INFO: LM Head loop finished. Time: {total_lm_head_time_ms:.2f} ms")
        # --- End LM Head Loop ---

        nlls_tensor = torch.cat(nlls)
        ppl = torch.exp(nlls_tensor.mean())
        model.config.use_cache = use_cache
        logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")

        # --- Timing Report ---
        total_inference_time_ms = total_layer_processing_time_ms + total_lm_head_time_ms
        total_inference_time_ms_cpu = total_layer_processing_time_ms_cpu + total_lm_head_processing_time_cpu
        if total_tokens_processed > 0:
            time_per_token_ms = total_inference_time_ms / total_tokens_processed
            time_per_token_ms_cpu = total_inference_time_ms_cpu / total_tokens_processed
            print(f"Total Inference Time (Layers + LM Head) (CPU): {total_inference_time_ms_cpu:.2f} ms")
            print(f"Average Inference Time per Token (CPU): {time_per_token_ms_cpu:.4f} ms/token")
            print(f"Total Inference Time (Layers + LM Head): {total_inference_time_ms:.2f} ms")
            print(f"Average Inference Time per Token: {time_per_token_ms:.4f} ms/token")
            list_total_inference_time.append(total_inference_time_ms)
            list_total_inference_time_perf_count.append(total_inference_time_ms_cpu)
            list_time_per_token.append(time_per_token_ms)
            list_time_per_token_perf_count.append(time_per_token_ms_cpu)
            list_ppl.append(ppl.item())
        else:
            print("No tokens processed, cannot calculate time per token.")
        # --- End Timing Report ---


    avg_total_inference_time = sum(list_total_inference_time) / len(list_total_inference_time)
    var_total_inference_time = sum(
        [(x - avg_total_inference_time) ** 2 for x in list_total_inference_time]
    ) / len(list_total_inference_time)
    avg_total_inference_time_perf_count = sum(list_total_inference_time_perf_count) / len(list_total_inference_time_perf_count)
    var_total_inference_time_perf_count = sum(
        [(x - avg_total_inference_time_perf_count) ** 2 for x in list_total_inference_time_perf_count]
    ) / len(list_total_inference_time_perf_count)

    avg_time_per_token = sum(list_time_per_token) / len(list_time_per_token)
    var_time_per_token = sum(
        [(x - avg_time_per_token) ** 2 for x in list_time_per_token]
    ) / len(list_time_per_token)
    avg_time_per_token_perf_count = sum(list_time_per_token_perf_count) / len(list_time_per_token_perf_count)
    var_time_per_token_perf_count = sum(
        [(x - avg_time_per_token_perf_count) ** 2 for x in list_time_per_token_perf_count]
    ) / len(list_time_per_token_perf_count)

    avg_ppl = sum(list_ppl) / len(list_ppl)

    # --- Save Detailed Results to JSON if path is provided ---
    if hasattr(args, 'timing_output_path') and args.timing_output_path:
        print(f"INFO: Saving detailed timing results to {args.timing_output_path}")
        results_to_save = {
            "nb_runs": max_trials,
            "list_ppl": list_ppl,
            "avg_ppl": avg_ppl,
            "cuda_timing_ms": {
                "list_total_time": list_total_inference_time,
                "list_token_time": list_time_per_token,
                "avg_total_time": avg_total_inference_time,
                "var_total_time": var_total_inference_time,
                "avg_token_time": avg_time_per_token,
                "var_token_time": var_time_per_token,
            },
            "cpu_timing_ms": {
                "avg_total_time": avg_total_inference_time_perf_count,
                "var_total_time": var_total_inference_time_perf_count,
                "avg_token_time": avg_time_per_token_perf_count,
                "var_token_time": var_time_per_token_perf_count,
            }
        }
        try:
            with open(args.timing_output_path, 'w') as f:
                json.dump(results_to_save, f, indent=4, default=lambda o: '<not serializable>')
            print(f"INFO: Successfully saved timing results to {args.timing_output_path}")
        except Exception as e:
            print(f"ERROR: Could not save timing results to {args.timing_output_path}: {e}")
    # --- End JSON Saving ---

    print(f"Average Total Inference Time: {avg_total_inference_time:.2f} ms")
    print(f"Average Total Inference Time (CPU): {avg_total_inference_time_perf_count:.2f} ms")
    print(f"Variance Total Inference Time: {var_total_inference_time:.2f} ms")
    print()
    print(f"Average Time per Token: {avg_time_per_token:.4f} ms/token")
    print(f"Average Time per Token (CPU): {avg_time_per_token_perf_count:.4f} ms/token")
    print(f"Variance Time per Token: {var_time_per_token:.4f} ms/token")
    print()
    print(f"Average PPL: {avg_ppl:.3f}")
    return avg_ppl, avg_time_per_token


@torch.no_grad()
def evaluator_single_gpu_simplified(model, testenc, dev, args):
    """
    Simple perplexity evaluation on a single GPU.
    
    Args:
        model: The language model to evaluate
        testenc: Test encoding (should have .input_ids attribute)
        dev: Device to run on
        args: Arguments containing batch_size and nb_eval_runs
    
    Returns:
        avg_ppl: Average perplexity across runs
        avg_time_per_token: Average inference time per token
    """
    dev = model.device
    
    print(f"INFO: Running {args.nb_eval_runs} evaluation passes")
    
    list_ppl = []
    list_time_per_token = []
    
    nsamples, seq_len = testenc.dataset.shape
    total_tokens = nsamples * seq_len
    
    for run in range(args.nb_eval_runs):
        # get nsamples and seq_len of testenc
        
        print(f"INFO: Evaluating {nsamples} samples with seq_len={seq_len}; Total tokens: {total_tokens}")
        
        # Compute loss and perplexity
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for batch_input_ids in tqdm(testenc, desc=f"(Eval Run {run+1}/{args.nb_eval_runs})"):
            # if device map is auto then dontt move input ids to model device, otherwise move to model device
            batch_input_ids = batch_input_ids.to(model.device)
            logits = model(batch_input_ids).logits  # (batch_size, seq_len, vocab_size)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
            shift_labels = batch_input_ids[:, 1:]  # (batch_size, seq_len-1)
            
            # Compute loss
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            nlls.append(loss.detach().cpu())
            
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        
        # Calculate perplexity
        nll = torch.cat(nlls).mean()
        ppl = torch.exp(nll)
        
        time_per_token_ms = elapsed_time_ms / total_tokens if total_tokens > 0 else 0
        
        list_ppl.append(ppl.item())
        list_time_per_token.append(time_per_token_ms)
        
        print(f"Run {run+1}: PPL={ppl.item():.3f}, Time/Token={time_per_token_ms:.4f} ms/token")
    
    # Calculate averages
    avg_ppl = sum(list_ppl) / len(list_ppl)
    avg_time_per_token = sum(list_time_per_token) / len(list_time_per_token)
    
    print(f"\nAverage PPL: {avg_ppl:.3f}")
    print(f"Average Time per Token: {avg_time_per_token:.4f} ms/token")
    torch.cuda.empty_cache()
    return avg_ppl, avg_time_per_token




# ---------------------------------------------------------------------------
# Core runner (original)
# ---------------------------------------------------------------------------

def run_evals(model, eval_tasks, device, batch_size=16):
    results = {}
    for task in eval_tasks:
        print(f"Evaluating on {task}...")
        try:
            task_results = lm_eval.evaluate(model, task, device=device, batch_size=batch_size)
            results[task] = task_results
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            results[task] = {"error": str(e)}
    return results


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _evaluate_task(model, task_name, device, batch_size=16, num_fewshot=0, limit=None, extra_kwargs=None):
    """
    Thin wrapper around lm_eval.simple_evaluate (harness v0.4+).
    Falls back to lm_eval.evaluate for older harness versions.
    Returns the per-task metrics dict.

    Parameters
    ----------
    limit : int or float or None
        Cap the number of samples evaluated per task.
        int   -> exact number of samples  (e.g. limit=100)
        float -> fraction of the dataset  (e.g. limit=0.1  means 10 %)
        None  -> use the full dataset (default)
    """
    kwargs = dict(
        model=model,
        tasks=[task_name],
        device=device,
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        **({"limit": limit} if limit is not None else {}),
        **(extra_kwargs or {}),
    )
    try:
        # lm_eval >= 0.4 (EleutherAI harness)
        output = lm_eval.simple_evaluate(**kwargs)
        result = output["results"][task_name]
        
        # destroy the full output dict immediately — it holds internal model refs
        del output
        gc.collect()
        torch.cuda.empty_cache()
        return result
    except AttributeError as e:
        # Older API
        print(e)
        return lm_eval.evaluate(model, task_name, device=device, batch_size=batch_size)


# ---------------------------------------------------------------------------
# OpenBookQA  (Openb.)
# ---------------------------------------------------------------------------

def eval_openbookqa(model, device, batch_size=16, num_fewshot=0, limit=None):
    """
    Evaluates on OpenBookQA (500 test questions, 4-way multiple-choice,
    elementary science facts).

    Key metric: acc_norm  (length-normalised accuracy, standard for this task)

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, acc_norm, (optionally) acc_stderr, acc_norm_stderr
    """
    print("=== OpenBookQA ===")
    result = _evaluate_task(
        model, "openbookqa", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc      = result.get("acc,none",      result.get("acc"))
    acc_norm = result.get("acc_norm,none", result.get("acc_norm"))
    print(f"  acc      : {acc:.4f}")
    print(f"  acc_norm : {acc_norm:.4f}")
    return result


# ---------------------------------------------------------------------------
# ARC-Easy  (ARC_e)
# ---------------------------------------------------------------------------

def eval_arc_easy(model, device, batch_size=16, num_fewshot=0, limit=None):
    """
    Evaluates on ARC-Easy (2 376 test questions, 4-way multiple-choice,
    grade-school science; the easier partition of the ARC corpus).

    Key metric: acc_norm

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, acc_norm, (optionally) stderr variants
    """
    print("=== ARC-Easy ===")
    result = _evaluate_task(
        model, "arc_easy", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc      = result.get("acc,none",      result.get("acc"))
    acc_norm = result.get("acc_norm,none", result.get("acc_norm"))
    print(f"  acc      : {acc:.4f}")
    print(f"  acc_norm : {acc_norm:.4f}")
    return result


# ---------------------------------------------------------------------------
# Winogrande  (WinoG.)
# ---------------------------------------------------------------------------

def eval_winogrande(model, device, batch_size=16, num_fewshot=5, limit=None):
    """
    Evaluates on Winogrande (1 267 test items, binary commonsense pronoun
    resolution).  Standard protocol uses 5-shot.

    Key metric: acc  (no normalisation needed - both continuations same length)

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, (optionally) acc_stderr
    """
    print("=== Winogrande ===")
    result = _evaluate_task(
        model, "winogrande", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc = result.get("acc,none", result.get("acc"))
    print(f"  acc : {acc:.4f}")
    return result


# ---------------------------------------------------------------------------
# HellaSwag  (HellaS.)
# ---------------------------------------------------------------------------

def eval_hellaswag(model, device, batch_size=16, num_fewshot=10, limit=None):
    """
    Evaluates on HellaSwag (10 042 validation items, 4-way sentence completion
    for activity descriptions).  Standard protocol uses 10-shot.

    Key metric: acc_norm  (length-normalised, strongly preferred here because
    the wrong continuations are adversarially length-matched)

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, acc_norm, (optionally) stderr variants
    """
    print("=== HellaSwag ===")
    result = _evaluate_task(
        model, "hellaswag", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc      = result.get("acc,none",      result.get("acc"))
    acc_norm = result.get("acc_norm,none", result.get("acc_norm"))
    print(f"  acc      : {acc:.4f}")
    print(f"  acc_norm : {acc_norm:.4f}")
    return result


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------

def eval_piqa(model, device, batch_size=16, num_fewshot=0, limit=None):
    """
    Evaluates on PIQA (1 838 test items, binary physical intuition QA).

    Key metric: acc_norm

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, acc_norm, (optionally) stderr variants
    """
    print("=== PIQA ===")
    result = _evaluate_task(
        model, "piqa", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc      = result.get("acc,none",      result.get("acc"))
    acc_norm = result.get("acc_norm,none", result.get("acc_norm"))
    print(f"  acc      : {acc:.4f}")
    print(f"  acc_norm : {acc_norm:.4f}")
    return result

def eval_truthfulqa(model, device, batch_size=16, num_fewshot=0, limit=None):
    """
    Evaluates on TruthfulQA (817 questions, multiple-choice format testing
    whether models avoid common misconceptions and falsehoods).

    Uses the 'truthfulqa_mc2' task from lm-eval-harness, which is the standard
    multiple-choice variant with multiple correct answers.

    Key metric: acc (average probability mass on true answers)

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, (optionally) acc_stderr
    """
    print("=== TruthfulQA ===")
    result = _evaluate_task(
        model, "truthfulqa_mc2", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc = result.get("acc,none", result.get("acc"))
    print(f"  acc : {acc:.4f}")
    return result

# ---------------------------------------------------------------------------
# MathQA
# ---------------------------------------------------------------------------

def eval_mathqa(model, device, batch_size=16, num_fewshot=0, limit=None):
    """
    Evaluates on MathQA (2 985 test items, 5-way multiple-choice covering
    arithmetic and algebraic word problems).

    Key metric: acc_norm

    Parameters
    ----------
    limit : int or float or None
        Cap samples (int) or fraction of dataset (float). None = full dataset.

    Returns
    -------
    dict with keys: acc, acc_norm, (optionally) stderr variants
    """
    print("=== MathQA ===")
    result = _evaluate_task(
        model, "mathqa", device, batch_size, num_fewshot=num_fewshot, limit=limit
    )
    acc      = result.get("acc,none",      result.get("acc"))
    acc_norm = result.get("acc_norm,none", result.get("acc_norm"))
    print(f"  acc      : {acc:.4f}")
    print(f"  acc_norm : {acc_norm:.4f}")
    return result


# ---------------------------------------------------------------------------
# Convenience: run all six benchmarks at once
# ---------------------------------------------------------------------------

BENCHMARK_FNS = {
    "truthfulqa": eval_truthfulqa,
    "openbookqa": eval_openbookqa,
    "arc_easy":   eval_arc_easy,
    "winogrande": eval_winogrande,
    "piqa":       eval_piqa,
    "hellaswag":  eval_hellaswag,
    # "mathqa":     eval_mathqa,
}

def run_standard_benchmarks(model, tokenizer, device, batch_size=32, limit=None):
    """
    Run all six standard benchmarks and return a consolidated results dict.

    Parameters
    ----------
    limit : int or float or None
        Cap samples per task for quick smoke-test runs.
        e.g. limit=100  -> at most 100 samples per benchmark
             limit=0.1  -> 10 % of each benchmark's test set
             limit=None -> full datasets (default)

    Returns
    -------
    {
        "openbookqa": {...},
        "arc_easy":   {...},
        "winogrande": {...},
        "hellaswag":  {...},
        "piqa":       {...},
        "mathqa":     {...},
    }
    """
    lm = MyCustomLM(model=model, tokenizer=tokenizer, device=device)
    results = {}
    for name, fn in BENCHMARK_FNS.items():
        try:
            results[name] = fn(lm, device, batch_size=batch_size, limit=limit)
        except Exception as e:
            print(f"Error on {name}: {e}")
            results[name] = {"error": str(e)}    
    print(results)
    # explicitly destroy the lm wrapper before returning
    del lm
    return results


BENCHMARK_SUBSET = {
    "openbookqa": eval_openbookqa,
    "arc_easy":   eval_arc_easy,
    "winogrande": eval_winogrande,
    "piqa":       eval_piqa,
    # "hellaswag":  eval_hellaswag,
    # "mathqa":     eval_mathqa,
}
def run_subset_benchmarks(model, tokenizer, device, batch_size=32, limit=None):
    """
    Run a subset of benchmarks (e.g. just OpenBookQA and ARC-Easy) for quick testing.

    Parameters
    ----------
    limit : int or float or None
        Cap samples per task for quick smoke-test runs.
        e.g. limit=100  -> at most 100 samples per benchmark
             limit=0.1  -> 10 % of each benchmark's test set
             limit=None -> full datasets (default)

    Returns
    -------
    {
        "openbookqa": {...},
        "arc_easy":   {...},
    }
    """
    lm = MyCustomLM(model=model, tokenizer=tokenizer, device=device)
    results = {}
    for name, fn in BENCHMARK_SUBSET.items():
        try:
            results[name] = fn(lm, device, batch_size=batch_size, limit=limit)
        except Exception as e:
            print(f"Error on {name}: {e}")
            results[name] = {"error": str(e)}    
    print(results)
    # explicitly destroy the lm wrapper before returning
    del lm
    return results