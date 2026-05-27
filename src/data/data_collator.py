import torch
import multiprocessing as mp
import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, NewType, Optional, Union

import numpy as np

from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
from transformers.data.data_collator import _torch_collate_batch

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator for language modeling that allows for pre-computed labels.

    This collator is a slight modification of `transformers.DataCollatorForLanguageModeling`.
    The key difference is in how it handles the `labels` for the language modeling task.

    - By default, `DataCollatorForLanguageModeling` creates labels by duplicating the `input_ids`.
    - This custom collator checks if a `labels` field already exists in the input features.
    - If `labels` are provided in a feature, those labels are used directly for training.
    - If `labels` are not provided, it falls back to the default behavior of duplicating `input_ids`.

    This is useful for scenarios where the labels are not identical to the input IDs,
    for example, in tasks that combine language modeling with other objectives or use a
    different tokenization scheme for labels.

    Note: This implementation is for PyTorch (`return_tensors='pt'`).
    """

    def __post_init__(self):
        super().__post_init__()
        # Ensure this collator is used with PyTorch tensors.
        if self.return_tensors != "pt":
            raise ValueError(
                "This custom data collator is only implemented for PyTorch (`return_tensors='pt'`)."
            )

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.

        if self.seed and self.generator is None:
            # If we have a seed, we need to create a generator object. Subsequent calls to this function will use the same generator.
            # If no seed supplied, we will use the global RNG
            self.create_rng()

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            if "labels" in batch.keys():
                labels = batch["labels"]
            else:                
                labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    # def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> tuple[Any, Any]:
    #     """
    #     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    #     """
    #     import torch

    #     labels = inputs.clone()
    #     # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    #     probability_matrix = torch.full(labels.shape, self.mlm_probability)
    #     if special_tokens_mask is None:
    #         special_tokens_mask = [
    #             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    #         ]
    #         special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    #     else:
    #         special_tokens_mask = special_tokens_mask.bool()

    #     probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    #     masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()
    #     labels[~masked_indices] = -100  # We only compute loss on masked tokens

    #     # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = (
    #         torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob), generator=self.generator).bool()
    #         & masked_indices
    #     )
    #     inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    #     if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
    #         return inputs, labels

    #     remaining_prob = 1 - self.mask_replace_prob
    #     # scaling the random_replace_prob to the remaining probability for example if
    #     # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
    #     # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
    #     random_replace_prob_scaled = self.random_replace_prob / remaining_prob

    #     # random_replace_prob% of the time, we replace masked input tokens with random word
    #     indices_random = (
    #         torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled), generator=self.generator).bool()
    #         & masked_indices
    #         & ~indices_replaced
    #     )
    #     random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
    #     inputs[indices_random] = random_words[indices_random]

    #     # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
    #     return inputs, labels
