#!/usr/bin/python
# -*- coding: utf-8 -*-

# %%

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

import stationary_reversal as sr
from reverse_sampling import *
from utils import create_dataset, create_chunked_dataset_from_full_sequences


# The memory usage of this function is dominated by
# the output of the model, (batch_size, sample_length, vocab_size).
# The default values here correspond to 5.24 gigabytes of memory.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')

    # Dataloader Parameters
    parser.add_argument('--dataset_name', type=str, default='pile_val')
    parser.add_argument('--num_examples', type=int, default=10,
        help='Number of examples to run loss over.'
    )
    parser.add_argument('--full_data_set_chunk', type=bool, default=True)
    parser.add_argument('--prefix_length', type=int, default=10,
        help='Number of tokens to predict in each example.'
    )
    parser.add_argument('--suffix_length', type=int, default=1,
        help='Context length for each example.'
    )
    parser.add_argument('--num_buffer', type=int, required=False,
        help='Where to begin the prefix.'
    )
    parser.add_argument('--batch_size', type=int, default=1,
        help='Batch size for loss calculation (i.e. number of suffixes).'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed.'
    )


    # Reversal Parameters    
    parser.add_argument('--vocab_batch_size', type=int, default=786,
        help='Number of words to batch when computing reverse probability.'
    )    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )    
    parser.add_argument('--dist', type=str, required=False,
        help='Path to the distribution file'
    )
    parser.add_argument('--dilution', type=float, default=0.0,
        help='dist = (1 - dilution) * dist + dilution * uniform'
    )
    parser.add_argument('--reverse_model_prior', type=bool, default=False,
        help='Use the reverse model as a prior')
    parser.add_argument('--multiple_priors_start_idx', type=int, default=0)
    parser.add_argument('--multiple_priors_end_idx', type=int, default=0)
    parser.add_argument('--model_size', type=str, default='160m')


    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(args.device)
    if device == 'cuda':
        print('Using gpu.')

    model_name = 'EleutherAI/pythia-' + args.model_size + '-deduped'

    if args.reverse_model_prior:
        reverse_model = GPTNeoXForCausalLM.from_pretrained(
            "afterless/reverse-pythia-160m"
        ).to(device)
    else:
        empirical_dist = torch.load(args.dist)
        if args.multiple_priors_end_idx > 0:
            empirical_dist = empirical_dist[:,args.multiple_priors_start_idx:args.multiple_priors_end_idx]
    
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

    if args.full_data_set_chunk:
        dataloader = create_chunked_dataset_from_full_sequences(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            batch_size=args.batch_size,
            seed=args.seed
        )
    else:
        dataloader = create_dataset(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            num_buffer=args.num_buffer,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name,
        revision="step3000",
        device_map="auto"
    ).to(device)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id) 
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing loss'):
            input_ids = batch['input_ids'].to(device)

            if args.reverse_model_prior:
                loss = compute_loss_reverse_dynamics_reverse_prior(
                    model,
                    reverse_model,
                    input_ids,
                    vocab_batch_size=args.vocab_batch_size,
                    dilution=args.dilution,  
                    device=device,
                    loss = criterion
                )
            else:
                loss = compute_loss_reverse_dynamics(
                    model,
                    empirical_dist,
                    input_ids,
                    vocab_batch_size=args.vocab_batch_size,
                    dilution=args.dilution,  
                    device=device,
                    loss = criterion
                )
            losses.append(loss)                                    
            

    loss_array = np.array(losses)
    loss_mean = np.mean(loss_array)
    loss_variance = np.var(loss_array)
    nbatches = len(dataloader)

    data = {
        'name': "stationary_reversal",
        'dataset' : args.dataset_name,
        'mean': loss_mean,
        'variance': loss_variance,
        'std_on_mean': np.std(loss_array) / np.sqrt(nbatches),
        'nbatches': nbatches,
    }
    args_dict = vars(args)
    data.update(args_dict)

    directory = 'data/' + args.dataset_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    dict_str = json.dumps(data, sort_keys=True)
    hash_obj = hashlib.md5(dict_str.encode())

    with open(f"{directory}/stationary-reversal-{args.model_size}-{hash_obj.hexdigest()}.json", 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()