#!/bin/bash

#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=devlab
#SBATCH --mem-per-cpu=10G

input_dir=$1

python -m to_fairseq.code_bpe_encoder \
    --merge-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/merges.txt \
    --vocab-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/vocab.json \
    --pretokenizer-split-newlines-only \
    --input-dirs ${input_dir} \
    --output-dir /checkpoint/dpf/data/processed_filenames_redact_2 \
    --use-hf-tokenizer \
    --splits-dir /checkpoint/dpf/data/splits/ \
    --metadata dstars source extension filename \
    --workers 40

# python -m to_fairseq.code_bpe_encoder \
#     --merge-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/merges.txt \
#     --vocab-file /checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/vocab.json \
#     --pretokenizer-split-newlines-only \
#     --input-dirs ${input_dir} \
#     --output-dir /checkpoint/dpf/data/processed_filenames_redact_d-0.1 \
#     --bpe-dropout 0.1 \
#     --use-hf-tokenizer \
#     --splits-dir /checkpoint/dpf/data/splits/ \
#     --metadata dstars source extension filename \
#     --workers 40
