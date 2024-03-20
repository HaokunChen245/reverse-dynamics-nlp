import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from transformers.generation.logits_process import (LogitsProcessor,
                                                    LogitsProcessorList)
from datasets import load_dataset
from typing import Callable, Iterable, Any
import matplotlib.pyplot as plt


SOFTMAX_FINAL = nn.Softmax(dim=-1)
LOGSOFTMAX_FINAL = nn.LogSoftmax(dim=-1)
CROSSENT = nn.CrossEntropyLoss(reduction='none')


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    embed_weights = list(model.modules())[2]
    assert type(embed_weights).__name__=='Embedding'
    embed_weights = embed_weights.weight
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    return one_hot.grad.clone()


class GreedyCoordinateGradient:
    """
    Implementation of Default GCG method, using the default
    settings from the paper (https://arxiv.org/abs/2307.15043v1)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n_proposals: int = 128,
        n_epochs: int = 100,
        n_top_indices: int = 128,
        prefix_loss_weight: float = 0.0,
        temperature: int = 0,
        revert_on_loss_increase: bool = True,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.n_proposals = n_proposals
        self.n_epochs = n_epochs
        self.n_top_indices = n_top_indices
        self.prefix_loss_weight = prefix_loss_weight
        self.temperature = temperature
        self.revert_on_loss_increase = revert_on_loss_increase

    def calculate_restricted_subset(
        self,
        input_ids,
        input_slice,
        target_slice,
        loss_slice
    ):
        # Find the subset of tokens that have the most impact on the likelihood of the target
        grad = token_gradients(self.model, input_ids, input_slice, target_slice, loss_slice)
        top_indices = torch.topk(-grad, self.n_top_indices, dim=-1).indices
        top_indices = top_indices.detach().cpu().numpy()
        return top_indices

    def sample_proposals(
        self,
        input_ids,
        top_indices,
        input_slice,
        target_slice,
        loss_slice,
    ):
        # Sample random proposals
        proposals = []
        if self.temperature:
            logits = self.model(input_ids.view(1, *input_ids.shape)).logits
            probs = SOFTMAX_FINAL(logits/self.temperature)
        for p in range(self.n_proposals):
            if self.temperature:
                token_pos = np.random.randint(input_slice.start, input_slice.stop)
                rand_token = torch.multinomial(probs[0, token_pos, :], 1).item()
            else:
                token_pos = np.random.randint(input_slice.start, input_slice.stop)
                rand_token = np.random.choice(top_indices[token_pos])
            prop = input_ids.clone()
            prop[token_pos] = rand_token
            proposals.append(prop)
        return torch.stack(proposals)

    def optimize(
        self,
        initial_input,
        target_string,
    ):
        # Parse input strings into tokens
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt")[0].cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt")[0].cuda()
        input_ids = torch.cat([initial_inputs, initial_targets], dim=0)
        input_slice = slice(0, initial_inputs.shape[0])
        target_slice = slice(initial_inputs.shape[0], input_ids.shape[-1])
        loss_slice = slice(initial_inputs.shape[0] - 1, input_ids.shape[-1] - 1)
        # Shifted input slices for prefix loss calculation
        shifted1 = slice(0, initial_inputs.shape[0] - 1)
        shifted2 = slice(1, initial_inputs.shape[0])
        # Optimize input
        prev_loss = None
        for i in range(self.n_epochs):
            # Get proposals for next string
            top_indices = self.calculate_restricted_subset(input_ids, input_slice, target_slice, loss_slice)
            proposals = self.sample_proposals(input_ids, top_indices, input_slice, target_slice, loss_slice)
            # Choose the proposal with the lowest loss
            with torch.no_grad():
                prop_logits = self.model(proposals).logits
                targets = input_ids[target_slice]
                losses = [nn.CrossEntropyLoss()(prop_logits[pidx, loss_slice, :], targets).item() for pidx in range(prop_logits.shape[0])]
                # Add a penalty for unlikely prompts that are not very high-likelihood
                if self.prefix_loss_weight > 0:
                    shifted_inputs = input_ids[shifted2]
                    prefix_losses = [nn.CrossEntropyLoss()(prop_logits[pidx, shifted1, :], shifted_inputs).item() for pidx in range(prop_logits.shape[0])]
                    losses = [losses[i] + self.prefix_loss_weight * prefix_losses[i] for i in range(len(losses))]
                # Choose next prompt
                new_loss = min(losses)
                min_idx = np.array(losses).argmin()
                #print(new_loss)
                if prev_loss is None or new_loss < prev_loss or not self.revert_on_loss_increase:
                    input_ids = proposals[min_idx]
                    prev_loss = new_loss
        return self.tokenizer.decode(input_ids)

