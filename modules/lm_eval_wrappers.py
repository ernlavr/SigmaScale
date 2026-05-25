import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from typing import List, Tuple
from tqdm import tqdm


class MyCustomLM(LM):
    def __init__(self, model, tokenizer, batch_size=1, device="cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size
        self.device = device

        # Fix missing pad_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    #  loglikelihood: P(continuation | context)                           #
    #  Input:  list of Instance with .args = (context_str, continuation_str)
    #  Output: list of (float logprob, bool is_greedy)                   #
    # ------------------------------------------------------------------ #
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        results = []

        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]
            batch_results = self._loglikelihood_batch(batch)
            results.extend(batch_results)

        return results

    def _loglikelihood_batch(self, batch: List[Instance]) -> List[Tuple[float, bool]]:
        results = []

        for instance in batch:
            context, continuation = instance.args

            full_text   = context + continuation
            full_ids    = self.tokenizer.encode(full_text, add_special_tokens=False)
            ctx_ids     = self.tokenizer.encode(context,   add_special_tokens=False)
            cont_ids    = full_ids[len(ctx_ids):]  # reliable boundary
            input_ids   = torch.tensor([full_ids]).to(self.model.device)

            with torch.no_grad():
                logits = self.model(input_ids, use_cache=False).logits  # (1, seq_len, vocab)

            # Shift: logits[i] predicts token[i+1]
            # We only care about the continuation slice
            cont_start = len(ctx_ids)                        # first cont token index
            cont_len   = len(cont_ids)

            # logits that predict continuation tokens
            shift_logits = logits[0, cont_start - 1 : cont_start - 1 + cont_len]  # (cont_len, vocab)
            shift_labels = input_ids[0, cont_start : cont_start + cont_len]        # (cont_len,)

            log_probs = F.log_softmax(shift_logits, dim=-1)  # (cont_len, vocab)

            # Sum log-probs over continuation tokens
            token_log_probs = log_probs[
                torch.arange(cont_len, device=self.device), shift_labels
            ]
            total_log_prob = token_log_probs.sum().item()

            # is_greedy: check if every continuation token was the argmax
            greedy_tokens = shift_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == shift_labels).all().item()

            results.append((total_log_prob, bool(is_greedy)))

        return results

    # ------------------------------------------------------------------ #
    #  generate_until: free generation with stop sequences                #
    #  Input:  list of Instance with .args = (context_str, gen_kwargs)   #
    #  Output: list of generated strings (continuation only)             #
    # ------------------------------------------------------------------ #
    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        for i in range(0, len(requests), self._batch_size):
            batch = requests[i : i + self._batch_size]
            batch_results = self._generate_batch(batch)
            results.extend(batch_results)

        return results

    def _generate_batch(self, batch: List[Instance]) -> List[str]:
        results = []

        for instance in batch:
            context, gen_kwargs = instance.args

            # Common gen_kwargs keys from lm_eval: until, max_gen_toks, temperature, do_sample
            stop_sequences = gen_kwargs.get("until", [])
            max_new_tokens  = gen_kwargs.get("max_gen_toks", 256)
            temperature      = gen_kwargs.get("temperature", 1.0)
            do_sample        = gen_kwargs.get("do_sample", False)

            input_ids = self.tokenizer.encode(
                context, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (strip the prompt)
            generated = self.tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )

            # Truncate at the first stop sequence
            for stop in stop_sequences:
                if stop in generated:
                    generated = generated[: generated.index(stop)]

            results.append(generated)

        return results

    # ------------------------------------------------------------------ #
    #  loglikelihood_rolling: P(sequence) with a sliding context window   #
    #  Input:  list of Instance with .args = (string,)                   #
    #  Output: list of float (total log-prob of the full string)          #
    # ------------------------------------------------------------------ #
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        results = []

        for instance in requests:
            string = instance.args[0]
            log_prob = self._rolling_logprob(string)
            results.append(log_prob)

        return results

    def _rolling_logprob(self, string: str) -> float:
        input_ids = self.tokenizer.encode(
            string, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        max_len   = self.model.config.max_position_embeddings  # model's context window
        seq_len   = input_ids.shape[1]
        total_log_prob = 0.0

        # Slide a window of size max_len across the sequence
        for start in range(0, seq_len - 1, max_len - 1):
            end     = min(start + max_len, seq_len)
            chunk   = input_ids[:, start:end]                 # (1, chunk_len)

            with torch.no_grad():
                logits = self.model(chunk).logits             # (1, chunk_len, vocab)

            # Shift for next-token prediction
            shift_logits = logits[0, :-1]                     # (chunk_len-1, vocab)
            shift_labels = chunk[0, 1:]                       # (chunk_len-1,)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[
                torch.arange(len(shift_labels), device=self.device), shift_labels
            ]

            # Skip the first token of subsequent chunks (it was the last of the prev window)
            skip = 1 if start > 0 else 0
            total_log_prob += token_log_probs[skip:].sum().item()

        return total_log_prob

    @property
    def batch_size(self):
        return self._batch_size
