# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility class and functions.

Adapted from:
https://github.com/kmeng01/rome/blob/bef95a6afd2ca15d794bdd4e3ee0f24283f9b996/
"""

import re

import torch
import transformers


class ModelAndTokenizer:
    """An object to hold a GPT-style language model and tokenizer."""

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        use_fast=True,
        device="cuda",
        local_model_path=None,
    ):
        if tokenizer is None:
            assert model_name is not None or local_model_path is not None, "Either model_name or local_model_path must be provided"
            path_to_use = local_model_path if local_model_path is not None else model_name
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                path_to_use, use_fast=use_fast
            )
        if model is None:
            assert model_name is not None or local_model_path is not None, "Either model_name or local_model_path must be provided"
            path_to_use = local_model_path if local_model_path is not None else model_name
            print(path_to_use)
            model = transformers.AutoModelForCausalLM.from_pretrained(
                path_to_use, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            if device is not None:
                model.to(device)
            set_requires_grad(False, model)
            model.eval()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = tokenizer.eos_token
        self.model = model
        self.device = device
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        """String representation of this class."""
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def make_inputs(tokenizer, prompts, device="cuda"):
    """Prepare inputs to the model.
    
    Args:
        tokenizer: The tokenizer to use for encoding the prompts.
        prompts: A list of text prompts to encode.
        device: The device to place the tensors on. Default is 'cuda'.
        
    Returns:
        A dictionary containing:
            - input_ids: Tensor of token IDs with padding.
            - attention_mask: Tensor indicating which tokens to attend to (1) vs padding (0).
    """
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    """Convert token IDs back to their string representations.
    
    Args:
        tokenizer: The tokenizer to use for decoding.
        token_array: Either a single token ID, a 1D array of token IDs, or a 2D array of token IDs.
        
    Returns:
        If token_array is a single token or 1D array, returns a list of decoded tokens.
        If token_array is a 2D array, returns a list of lists of decoded tokens.
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    """Find the tokens corresponding to the given substring in token_array.
    
    Args:
        tokenizer: The tokenizer used for decoding.
        token_array: An array of token IDs.
        substring: The string to locate within the decoded tokens.
        
    Returns:
        A tuple (start_index, end_index) indicating the token indices that
        contain the substring. The range is inclusive at the start and
        exclusive at the end.
    """
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_from_input(model, inp):
    """
    Generate predictions from input using a given model.

    This function computes the forward pass of the model with the given input,
    applies softmax to the last token's logits to get probabilities,
    and returns the highest probability prediction along with its probability value.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction, which should return a dictionary with a 'logits' key
        when called with the input.
    inp : dict
        Input dictionary to be passed to the model. This should contain all the necessary
        tensors required by the model.

    Returns
    -------
    tuple
        A tuple containing:
        - preds (torch.Tensor): The predicted token IDs (argmax of the probability distribution)
        - p (torch.Tensor): The probability values associated with the predictions
    """
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def set_requires_grad(requires_grad, *models):
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)
