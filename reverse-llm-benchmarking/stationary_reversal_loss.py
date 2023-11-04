#!/usr/bin/python
# -*- coding: utf-8 -*-

# %%

import argparse
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from einops import rearrange
from datasets import concatenate_datasets
import numpy as np
import json
import os

from reverse_sampling import *

#import sys
#sys.path.append('../stationary_reversal.py')

#import stationary_reversal as sr


# The memory usage of this function is dominated by
# the output of the model, (batch_size, sample_length, vocab_size).
# The default values here correspond to 5.24 gigabytes of memory.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')

    parser.add_argument('--samples', type=int, default=10,
        help='Number of samples to keep.'
    )
    
    parser.add_argument('--suffix_batch_size', type=int, default=1,
        help='Batch size for loss calculation (i.e. number of suffixes).'
    )

    parser.add_argument('--sample_length', type=int, default=10,
        help='Where to truncate the input sequences.'
    )

    parser.add_argument('--vocab_batch_size', type=int, default=512,
        help='Number of words to batch when computing reverse probability.'
    )
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )
    
    parser.add_argument('--dist', type=str, required=True,
        help='Path to the distribution file'
    )
    
    parser.add_argument('--dillution', type=float, default=0.0,
        help='dist = (1 - dillution) * dist + dillution * uniform'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(args.device)
    if device == 'cuda':
        print('Using gpu.')

    sample_size = args.samples
    suffix_batch_size = args.suffix_batch_size
    sample_length = args.sample_length
    vocab_batch_size = args.vocab_batch_size

    model_sizes = ['70m', '160m', '410m']
    model_names = ['EleutherAI/pythia-' + size + '-deduped-v0'
                   for size in model_sizes]

    list_of_dataset_names = ['pile_val']  # ["small-pile-dedup-train", "TinyStories"]

    empirical_dist = torch.load(args.distribution)
    uniform_dist = torch.ones_like(empirical_dist) / empirical_dist.shape[0]
    empirical_dist = empirical_dist * (1 - args.dillution) + uniform_dist * args.dillution

    #list_of_stationary_distributions = ['empirical_dist', 'uniform_dist', 'Markov_stationary_dist']

    for dataset_name in list_of_dataset_names:
        if dataset_name == 'small-pile-dedup-train':

            # Using the Pile
            dataset = load_dataset('ola13/small-the_pile-dedup')

            # Concatenate all datasets in the DatasetDict
            concatenated_dataset = concatenate_datasets([ds for ds in dataset.values()])

            # Shuffle the concatenated dataset
            shuffled_dataset = concatenated_dataset.shuffle(seed=42)  # You can set your desired seed

            sampled_dataset = shuffled_dataset.select(range(sample_size))
            dataset = sampled_dataset

        elif dataset_name == 'TinyStories':

          # Using the Tiny Stories Data Set
            dataset = load_dataset('roneneldan/TinyStories', split='validation')
            dataset = dataset.select(range(sample_size))

        elif dataset_name == 'pile_val':
            dataset = load_dataset('json', data_files='data/val.jsonl')
            dataset = dataset['train'].select(range(sample_size))

        for (model_name, model_size) in zip(model_names, model_sizes):
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision="step3000",
                device_map="auto"
            )
            tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

            tokenizer.pad_token = tokenizer.eos_token

            def tokenize_text(example):
                return {'input_ids': tokenizer.encode(example['text'],
                        truncation=True, padding='max_length',
                        max_length=sample_length, return_tensors='pt'
                        ).squeeze(0)}

            tokenized_dataset = dataset.map(tokenize_text)

            # Get all column names
            all_columns = tokenized_dataset.column_names

            # Find columns to remove
            columns_to_remove = [column for column in all_columns
                                 if column != 'input_ids']

            # Remove unwanted columns
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

            # Debug
            # first_array = np.array(tokenized_dataset["input_ids"])
            # # max_occuring_token = max(tokenized_dataset["input_ids"], key=tokenized_dataset["input_ids"].count)
            # num_zeros = np.count_nonzero(first_array == 0)
            # num_nonzeros = np.count_nonzero(first_array != 0)
            # print(num_zeros/(num_zeros + num_nonzeros))

            # Use DataCollator to handle padding during training
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                return_tensors='pt'
            )

            # Convert dataset to DataLoader for batch processing
            dataloader = DataLoader(tokenized_dataset, shuffle=True,
                                    collate_fn=data_collator,
                                    batch_size=suffix_batch_size)

            model.eval()
            criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # This is your loss function
            losses = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc='Computing loss'):
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['input_ids'][:, :-1].to(device)

                    print(input_ids.shape)

                    # I assume it is fine to cross entropy with logprobs versus logits it's all the same
                    '''
                    logits = \
                        sr.stationary_reverse_full_dist_suffix_calculation(model,
                            empirical_dist, input_ids,
                            vocab_batch_size=vocab_batch_size,
                            renormalize_dist=True)
                    '''
                    _, logits = sample_reverse_dynamics(
                        model=model,
                        stationary_dist=empirical_dist,
                        prefix_length=sample_length,
                        tokenized_suffix=input_ids,
                        vocab_batch_size=1024,
                        temperature=0.7,
                        device=device
                    )   

                    # logits = rearrange(logits, 'b n c -> (b n) c')
                    targets = rearrange(targets, 'b n -> (b n)')

                    loss = criterion(logits, targets)
                    losses.append(loss.item())

            loss_array = np.array(losses)
            loss_mean = np.mean(loss_array)
            loss_variance = np.var(loss_array)
            nbatches = len(dataloader)

            data = {
                'mean': loss_mean,
                'variance': loss_variance,
                'std_on_mean': np.std(loss_array) / np.sqrt(nbatches),
                'total_samples': sample_size,
                'suffix_batch_size': suffix_batch_size,
                'nbatches': nbatches,
                'sample_length': sample_length,
                }

            directory = 'data/' + dataset_name
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(directory + '/stationary-reversal-' + model_size
                      + '-loss' + '-samplelength-' + str(sample_length)
                      + '.json', 'w') as f:
                json.dump(data, f)


      # np.save(directory+"/stationary-reversal-" + model_size + "-loss-samples.npy", loss_array)

if __name__ == '__main__':
    main()
