import re
import random
import itertools
import lxml.etree
import tqdm
from datasets import load_dataset
from code_clippy_dataset.utils import load_dataset_infer

import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer

import glob as glob

MAX_DOC_LENGTH = 10_000

NEWLINE_REP = "<|n|>"

SPLIT_LINES = re.compile(f'.*[\r\n]+')

def glob_reader(glob_path, yield_rate=None):
    # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
    for fname in glob.glob(glob_path):
        with open(fname, 'r') as f:
            for line in f:
                if yield_rate is None or rng.random() <= yield_rate:
                    yield line
    data_dir = data_dir.rstrip("/")
    data = load_dataset_infer(data_dir)
    for x in tqdm.tqdm(data, ncols=80, desc=data_dir):
        if yield_rate is None or rng.random() <= yield_rate:
            yield x['text']

def preprocess_text(text, max_len=MAX_DOC_LENGTH):
    if replace_newline:
        i = 0
        while i < len(text):
            piece = text[i:i+max_len]
            piece = piece.replace("\n", NEWLINE_REP)
            if piece:
                yield piece
            i += max_len
    else:
        matches = re.findall(SPLIT_LINES, text)
        if not matches:
            yield text
        else:
            yield from iter(matches)

if __name__ == "__main__":
    import os
    import argparse
    import sys
    import pprint

    print(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_dir_file_limit", type=int, default=50000)
    parser.add_argument("--stackoverflow_subsample_rate", type=float, default=0.001)
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--tokenizer_type", choices=['sentencepiece', 'byte_level_bpe'], default='byte_level_bpe')
    parser.add_argument("--replace_newline", action='store_true')
    parser.add_argument("--model_prefix",  default="tokenizers/github-py+so")
    parser.add_argument("--bpe_pretokenizer_split_newlines_only",  action="store_true")
    args = parser.parse_args()
    pprint.pprint(vars(args))

    # directory and number of files to take
    data_dirs = [
        ("/private/home/dpf/data/github/python_forkless_open-source_2star+/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn/", args.per_dir_file_limit),
        ("/private/home/dpf/data/github/python_forkless_open-source_1star/data_dedup_filtered_mwcf-0.4_mll-3000_pandoc_csn/", args.per_dir_file_limit),
        #("/private/home/dpf/data/github/javascript_forkless_open-source/data_dedup_filtered/", args.per_dir_file_limit),
    ]

    # file and yield (subsampling) rate
    stack_exchange = [
        #("/private/home/dpf/projects/stackexchange_dataset/dumps/stackoverflow/Posts.xml", args.stackoverflow_subsample_rate),
        ("/scratch/dpf/data/stackexchange_dataset/dumps/stackoverflow/Posts.xml", args.stackoverflow_subsample_rate),
        #("/private/home/dpf/projects/stackexchange_dataset/dumps/stackoverflow/Comments.xml", args.stackoverflow_subsample_rate),
    ]
    
    model_prefix = args.model_prefix
    vocab_size = args.vocab_size

    rng = random.Random(1)

    generators = []
    for data_dir, limit in data_dirs:
        generators.append(itertools.islice(dataset_reader(data_dir, rng), limit))
    for path, yield_rate in stack_exchange:
        generators.append(stackexchange_reader(path, rng, yield_rate=yield_rate, parse_html=True))

    # limit = 2000
    # if limit is not None:
    #     generators = [itertools.islice(generator, limit) for generator in generators]
    #     model_prefix += "_small"

    replace_newline = args.replace_newline

    generator = iter(
        piece for generator in generators
        for text in generator
        # since \n gets expanded, add in some buffer room by making pre-expansion chunks substantially smaller than MAX_DOC_LEN
        for piece in preprocess_text(text, max_len=MAX_DOC_LENGTH // 3 if replace_newline else MAX_DOC_LENGTH, replace_newline=replace_newline)
    )

    if replace_newline:
        user_defined_symbols = [NEWLINE_REP]
    else:
        user_defined_symbols = []

    user_defined_symbols += ['<|endoftext|>', '<|pad|>', '<|mask|>']

    if args.tokenizer_type == "byte_level_bpe":
        tokenizer = ByteLevelBPETokenizer(pretokenizer_split_newlines_only=args.bpe_pretokenizer_split_newlines_only)
        tokenizer.train_from_iterator(generator, vocab_size=vocab_size+257, special_tokens=None)
        #model_dir = model_prefix+f"_bpe_rn-{replace_newline}"
        if args.replace_newline:
            raise NotImplementedError()
        model_dir = model_prefix+f"_psno-{args.bpe_pretokenizer_split_newlines_only}"
        os.makedirs(model_dir, exist_ok=True)
        tokenizer.save_model(model_dir)
    elif args.tokenizer_type == "sentencepiece":
        spm.SentencePieceTrainer.train(
            sentence_iterator=generator, model_prefix=model_prefix+f"_spm-rn-{replace_newline}", vocab_size=vocab_size,
            user_defined_symbols=user_defined_symbols,
            model_type='unigram',
            max_sentence_length=MAX_DOC_LENGTH,
            max_sentencepiece_length=128,
            split_by_unicode_script=False,
            split_by_number=False,
            split_by_whitespace=False,
            split_digits=False,
            remove_extra_whitespaces=False,
            train_extremely_large_corpus=True,
            normalization_rule_name="identity", # prevent mapping tabs to spaces, with side effect of not doing any other unicode normalization
        )
    else:
        raise NotImplementedError(args.tokenizer_type)
