#!/bin/bash
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true 
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution 0.2
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution 0.4

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution --model_size 410m

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 1.0 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 10 --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 20 --dist ../data/distributions/pile10k_empirical.pt

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 10 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 20 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.0 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.2 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.4 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9

python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.0 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.2 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.4 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9





