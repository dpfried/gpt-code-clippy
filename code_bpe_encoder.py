#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import os
import io
import re
import math
import random
import functools

import numpy as np
import scipy
import scipy.stats

import tqdm

import datasets

from code_clippy_dataset.utils import infer_setlanguage_from_data_dir, infer_source_from_data_dir, postprocess_ipynb, grouper

from gpt2_bpe_utils import get_encoder

STAR_THRESHOLDS = {
    # 0 <= x < 2; 2 <= x < 4; 4 <= x < 9; 9 <= x < 30
    ('github', 'javascript'): [0, 2, 4, 9, 30],
    ('github', 'jupyter'):    [0, 1, 2, 5, 19],
    ('github', 'python'):     [0, 1, 3, 8, 36],
    ('gitlab', 'javascript'): [0, 1, 3, float('inf'), float('inf')],
    ('gitlab', 'python'):     [0, 1, 2, float('inf'), float('inf')],
    ('google-code', 'javascript'): [0, 1, 2, float('inf'), float('inf')],
    ('google-code', 'python'): [0, 1, 2, 5, float('inf')],
}

def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merge-file",
        help="path to merges.txt",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        help="path to vocab.txt",
    )
    parser.add_argument(
        "--pretokenizer-split-newlines-only",
        action='store_true',
    )
    parser.add_argument(
        "--use-hf-tokenizer",
        action='store_true',
        help='use huggingface tokenizer. slower (because of multiprocessing constraints?) but to check correctness'
    )
    parser.add_argument(
        "--input-dirs",
        required=True,
        nargs="+",
        help="paths to directories containing serialized HF datasets",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory to save encoded outputs",
    )
    parser.add_argument(
        '--splits-dir',
         required=True,
    )

    parser.add_argument(
        '--instances-per-shard',
        type=int,
        default=500_000,
    )

    parser.add_argument(
        '--max-instances',
        type=int, 
    )

    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    def read_project_names(split):
        with open(os.path.join(args.splits_dir, split), 'r') as f:
            return set(l.strip().lower() for l in f.readlines())

    train_project_names = read_project_names('train')
    val_project_names = read_project_names('val')

    encoder = MultiprocessingEncoder(args, train_project_names, val_project_names)

    os.makedirs(os.path.join(args.output_dir, 'bpe'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'raw'), exist_ok=True)

    def save_to_shards(data, name, split_name, extension):
        for shard_num, chunk in enumerate(grouper(args.instances_per_shard, data)):
            fname = os.path.join(args.output_dir, extension, f'{split_name}_{name}_{shard_num}.{extension}')
            with open(fname, 'w') as f:
                for instance in chunk:
                    f.write(instance)
                    # document separator
                    f.write('\n\n')

    for input_dir in args.input_dirs:
        data = datasets.load_from_disk(input_dir)

        if args.max_instances:
            data = data[:args.max_instances]

        encoder.dataset_dir = input_dir

        dataset_name = '_'.join(input_dir.split('/')[-3:-1])
        print(f'{dataset_name}\t({len(data):_}): \t {input_dir}')

        data = data.add_column('dataset_dir', np.full(len(data), input_dir))
        data_augmented = data.map(encoder.process, num_proc=args.workers)

        train_data = data_augmented.filter(lambda record: record['split'] == 'train', num_proc=args.workers)
        val_data = data_augmented.filter(lambda record: record['split'] == 'val', num_proc=args.workers)

        save_to_shards(train_data['augmented_text'], dataset_name, 'train', 'raw')
        save_to_shards(val_data['augmented_text'], dataset_name, 'val', 'raw')

        pool = Pool(args.workers, initializer=encoder.initializer)

        train_tokenized = tqdm.tqdm(pool.imap(encoder.tokenize, train_data['augmented_text'], 100), total=len(train_data))
        save_to_shards(train_tokenized, dataset_name, 'train', 'bpe')

        val_tokenized = tqdm.tqdm(pool.imap(encoder.tokenize, val_data['augmented_text'], 100), total=len(val_data))
        save_to_shards(val_tokenized, dataset_name, 'val', 'bpe')


def make_tagged(tag, inner, attributes={}, insert_newlines=True, attribute_move_probability=None):
    if attributes:
        attr_strs = [f'{k}={v}' for k, v in attributes.items()]
        if attribute_move_probability is not None:
            assert 0 <= attribute_move_probability <= 1.0
            begin_attr_strs = []
            end_attr_strs = []
            for x in attr_strs:
                if random.random() < attribute_move_probability:
                    end_attr_strs.append(x)
                else:
                    begin_attr_strs.append(x)
        else:
            begin_attr_strs = attr_strs
            end_attr_strs = []
    else:
        begin_attr_strs = []
        end_attr_strs = []
    del attr_strs
    if begin_attr_strs:
        random.shuffle(begin_attr_strs)
        begin_attr_string = f" {' '.join(begin_attr_strs)}"
    else:
        begin_attr_string = ''
    if end_attr_strs:
        random.shuffle(end_attr_strs)
        end_attr_string = f" {' '.join(end_attr_strs)}"
    else:
        end_attr_string = ''
    if insert_newlines:
        return f'<| {tag}{begin_attr_string} |>\n{inner}\n<|/ {tag}{end_attr_string} |>'
    else:
        return f'<| {tag}{begin_attr_string} |> {inner} <|/ {tag}{end_attr_string} |>'

class MultiprocessingEncoder(object):

    def __init__(self, args, train_project_names, val_project_names):
        self.args = args
        # self.bpes = [self._make_bpe() for _ in range(self.args.workers)]
        self.train_project_names = train_project_names
        self.val_project_names = val_project_names

    def _make_bpe(self):
        if self.args.use_hf_tokenizer:
            from tokenizers import ByteLevelBPETokenizer
            bpe = ByteLevelBPETokenizer.from_file(
                self.args.vocab_file, self.args.merge_file,
                pretokenizer_split_newlines_only=self.args.pretokenizer_split_newlines_only,
                )
        else:
            pretokenizer_split = "newlines_only" if self.args.pretokenizer_split_newlines_only else "default"
            bpe = get_encoder(self.args.vocab_file, self.args.merge_file, pretokenizer_split)
        return bpe

    def process(self, example):
        dataset_dir = example['dataset_dir']
        source = infer_source_from_data_dir(dataset_dir)
        if source == 'bigquery':
            source = 'github'
        setlanguage = infer_setlanguage_from_data_dir(dataset_dir)

        ts = example['repo_name'].lower().split('/')
        if len(ts) != 2:
            project_name = '/'.join(ts)
        else:
            project_name = ts[-1]

        if project_name in self.train_project_names:
            split = 'train'
        elif project_name in self.val_project_names:
            split = 'val'
        else:
            raise ValueError(f"project_name {project_name} not found in train or val list!")

        attributes = {'source': source}

        stars = int(example.get('stars', '-1'))

        _, extension = os.path.splitext(example['file_name'])
        if extension is not None and extension.strip():
            if extension == '.ipynb':
                text, extension = postprocess_ipynb(example['text'])
            else:
                text = example['text']
            attributes['ext'] = extension

        if stars >= 0 and (source, setlanguage) in STAR_THRESHOLDS:
            thresholds = STAR_THRESHOLDS[(source, setlanguage)]
            disc_threshold = 0
            while disc_threshold < len(thresholds) and thresholds[disc_threshold] <= stars:
                disc_threshold += 1
            disc_threshold -=1
            assert 0 <= disc_threshold < len(thresholds)

            attributes['dstars'] = disc_threshold

        augmented_file = make_tagged('file', text, attributes=attributes, insert_newlines=True, attribute_move_probability=0.5)
        # encoded = self.encode_text(rank, augmented_file)

        return {
            'augmented_text': augmented_file,
            # 'augmented_text_tokens': encoded,
            'split': split
        }

    def initializer(self):
        global bpe
        bpe = self._make_bpe()

    def tokenize(self, text):
        global bpe
        if self.args.use_hf_tokenizer:
            ids = bpe.encode(text).ids
        else:
            ids = bpe.encode(text)
        return ' '.join(map(str, ids))


if __name__ == "__main__":
    main()