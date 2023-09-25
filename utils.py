import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from transformers.generation.logits_process import (LogitsProcessor,
                                                    LogitsProcessorList)
from datasets import load_dataset


SOFTMAX_FINAL = nn.Softmax(dim=-1)
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


def reverse_tokenize(tokenizer, target):
    input_ids = tokenizer.encode(target, return_tensors="pt").cuda()
    input_ids = torch.flip(input_ids, (1,))
    return input_ids


def reverse_output(output):
    return torch.flip(output, (1,))


def reverse_decode(tokenizer, output):
    tokens = torch.flip(output, (1,))
    return [
        tokenizer.decode(tokens[i]) for i in range(tokens.shape[0])
    ]


def reverse_generate(reverse_model, tokenizer, target, n_tokens):
    inputs = reverse_tokenize(tokenizer, target)
    outputs = reverse_model.generate(
        input_ids=inputs,
        max_new_tokens=n_tokens,
        do_sample=False
    )
    return reverse_decode(tokenizer, outputs)

def reverse_normalized_forward(reverse_model, tokenizer, target, token_statistics_vector):
    inputs = reverse_tokenize(tokenizer, target)
    outputs = reverse_model(inputs).logits
    outputs = SOFTMAX_FINAL(outputs)
    outputs = torch.mul(outputs, token_statistics_vector)
    outputs = torch.argmax(outputs, dim=-1)
    return reverse_decode(tokenizer, outputs)


class SampleTopTokens(LogitsProcessor):

    def __init__(self, n_initial_tokens, n_new_tokens, top_grad_tokens):
        self.n_initial_tokens = n_initial_tokens
        self.n_new_tokens = n_new_tokens
        self.top_grad_tokens = top_grad_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_pos = self.n_new_tokens - (input_ids.shape[-1] - self.n_initial_tokens) - 1
        mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
        mask[:, self.top_grad_tokens[curr_pos]] = False
        scores.masked_fill_(mask, -float('inf'))
        return scores

def get_token_probabilities(tokenizer, dataset="NeelNanda/pile-10k", vocab_size=50304):
    data = load_dataset(dataset)
    counts = torch.zeros(vocab_size, dtype=torch.float) #tokenizer.vocab_size is fake 50304 is the model output dimension which is what we care about

    for chunk in data['train']:
        # Extract text from chunk (assuming each chunk is a dictionary with a "text" key)
        text = chunk['text']

        # Tokenize the text
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]

        # Count occurrences for each token
        for tok in tokens:
            counts[tok] += 1

    # Normalize the counts to get probabilities
    total_tokens = torch.sum(counts)
    probabilities = counts / total_tokens
    min_val = probabilities[probabilities > 0].min()
    probabilities[probabilities == 0] = min_val
    return probabilities