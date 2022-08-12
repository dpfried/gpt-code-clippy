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

from to_fairseq.gpt2_bpe_utils import get_encoder

from to_fairseq.patch_pickle import patch_mp_connection_bpo_17560

STAR_THRESHOLDS = {
    ('github', 'javascript'): [2, 5, 15, 41, 102, 239],
    ('github', 'python'):     [0, 2, 8, 24, 66, 161],
    ('github', 'jupyter'):    [0, 1, 4, 12, 32, 84],
    # ('gitlab', 'javascript'): [0, 0, 0, 0, 0, 1], # too few repos have stars for this to be informative
    # ('gitlab', 'python'):     [0, 0, 0, 0, 1, 1], # too few repos have stars for this to be informative
    # ('google-code', 'javascript'): [0, 0, 0, 0, 1, 1], # too few repos have stars for this to be informative
    # ('google-code', 'python'): [0, 0, 0, 1, 2, 4], # too few repos have stars for this to be informative
}

# from https://github.com/madisonmay/CommonRegex/blob/2425abdb79c8992b8b655c27e1fb195cc54457ab/commonregex.py#L10, modified to take out some common delimiters to avoid replacing e.g. author='x@y.com'
EMAIL_RE = re.compile("([a-z0-9!#$%&*+?^_|.~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE)
# more permissive separators but needs to match more exact domain
EMAIL_RE_2 = re.compile("([a-z0-9!#$%&*+?^_|.~-]+((\s*@\s*|\s+at\s+|\s+a\s+|\s*\(at\)\s*|\s*\[at\]\s*|\s*\(a\)\s*))[a-z0-9.]+(\.|\s+dot\s+|\s*\(dot\)\s*|\s*\[dot\]\s*)(com|co\.[a-z][a-z]|ac\.[a-z][a-z]|edu|org|gov|net))", re.IGNORECASE)

def main():
    """
    Helper script to encode raw text with the BPE and write in a format that can be read by fairseq-preprocess, using multiple processes.
    Based on code from fairseq

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
        help='use huggingface tokenizer. faster than python implementation'
    )
    parser.add_argument(
        "--maxlen-tokenizer-merge-file",
        help="path to merges.txt",
    )
    parser.add_argument(
        "--maxlen-tokenizer-vocab-file",
        type=str,
        help="path to vocab.txt",
    )
    parser.add_argument(
        "--maxlen-tokens",
        type=int,
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
        help="path to directory containing 'train' and 'val' files with list of project names (lowercased repo names with username removed) to use in each split"
    )

    parser.add_argument(
        '--instances-per-shard',
        type=int,
        default=500_000,
    )

    parser.add_argument(
        '--attribute-move-probability',
        type=float,
        default=0.5,
    )

    parser.add_argument(
        '--metadata',
        nargs='*',
        choices=['dstars', 'source', 'extension', 'filename'],
        default=['dstars', 'source', 'extension'],
    )

    parser.add_argument(
        '--no-redact-pii',
        action='store_true',
    )

    parser.add_argument(
        '--bpe-dropout',
        type=float,
    )

    parser.add_argument("--workers", type=int, default=20)

    parser.add_argument("--file-extension-keep", nargs="*", help="keep only files ending in these extensions")
    parser.add_argument("--file-extension-exclude", nargs="*", help="exclude any files ending in these extensions")
    args = parser.parse_args()


    patch_mp_connection_bpo_17560()

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
        return shard_num + 1

    datasets.set_caching_enabled(False)

    for input_dir in args.input_dirs:
        data = datasets.load_from_disk(input_dir)

        encoder.dataset_dir = input_dir

        dataset_name = '_'.join(input_dir.rstrip('/').split('/')[-3:-1])
        print(f'{dataset_name}\t({len(data):_}): \t {input_dir}')

        data = data.add_column('dataset_dir', np.full(len(data), input_dir))
        data_augmented = data.map(encoder.process, num_proc=args.workers)

        if args.file_extension_keep is not None:
            assert args.file_extension_exclude is None, "can't pass both --file-extension-keep and --file-extension-exclude"
            assert all(ext.startswith(".") for ext in args.file_extension_keep)
            file_extension_keep = set(args.file_extension_keep)
            print(f"filtering to only include extensions {' '.join(sorted(file_extension_keep))}")
        else:
            file_extension_keep = None

        if args.file_extension_exclude is not None:
            assert args.file_extension_keep is None, "can't pass both --file-extension-keep and --file-extension-exclude"
            assert all(ext.startswith(".") for ext in args.file_extension_exclude)
            file_extension_exclude = set(args.file_extension_exclude)
            print(f"filtering to exclude any files with extensions in {' '.join(sorted(file_extension_exclude))}")
        else:
            file_extension_exclude = None

        def filter_function(split):
            def fn(record):
                if file_extension_keep is not None or file_extension_exclude is not None:
                    extension = os.path.splitext(record['file_name'])[1]
                    if file_extension_keep is not None and extension not in file_extension_keep:
                        return False
                    if file_extension_exclude is not None and extension in file_extension_exclude:
                        return False
                if record['over_max_length']:
                    return False
                return record['split'] == split
            return fn

        train_data = data_augmented.filter(filter_function('train'), num_proc=args.workers)
        val_data = data_augmented.filter(filter_function('val'), num_proc=args.workers)

        print(f"{dataset_name}: filtered")
        train_raw_shards = save_to_shards(train_data['augmented_text'], dataset_name, 'train', 'raw')
        print(f"{dataset_name}: saved train-raw in {train_raw_shards} shards")
        val_raw_shards = save_to_shards(val_data['augmented_text'], dataset_name, 'val', 'raw')
        print(f"{dataset_name}: saved val-raw in {val_raw_shards} shards")

        pool = Pool(args.workers, initializer=encoder.initializer)

        train_tokenized = tqdm.tqdm(pool.imap(encoder.tokenize, train_data['augmented_text'], 100), total=len(train_data))
        train_bpe_shards = save_to_shards(train_tokenized, dataset_name, 'train', 'bpe')
        print(f"{dataset_name}: saved train-bpe in {train_bpe_shards} shards")

        val_tokenized = tqdm.tqdm(pool.imap(encoder.tokenize, val_data['augmented_text'], 100), total=len(val_data))
        val_bpe_shards = save_to_shards(val_tokenized, dataset_name, 'val', 'bpe')
        print(f"{dataset_name}: saved val-bpe in {val_bpe_shards} shards")

def redact_pii(text):
    #text = EMAIL_RE.sub("removed@example.com", text)
    text = EMAIL_RE_2.sub("removed@example.com", text)
    return text

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
        if self.args.maxlen_tokens is not None:
            assert self.args.use_hf_tokenizer
            # TODO: maybe make this explicitly multiprocess as well, or maybe it's fine because of forks
            from tokenizers import ByteLevelBPETokenizer
            self.maxlen_bpe = ByteLevelBPETokenizer.from_file(
                self.args.maxlen_tokenizer_vocab_file, self.args.maxlen_tokenizer_merge_file,
                pretokenizer_split_newlines_only=False,
                )
        else:
            self.maxlen_bpe = None

    def _make_bpe(self):
        if self.args.use_hf_tokenizer:
            from tokenizers import ByteLevelBPETokenizer
            if self.args.bpe_dropout:
                print(f"dropout: {self.args.bpe_dropout}")
            bpe = ByteLevelBPETokenizer.from_file(
                self.args.vocab_file, self.args.merge_file,
                pretokenizer_split_newlines_only=self.args.pretokenizer_split_newlines_only,
                dropout=self.args.bpe_dropout,
                )
        else:
            assert self.args.bpe_dropout == None
            pretokenizer_split = "newlines_only" if self.args.pretokenizer_split_newlines_only else "default"
            bpe = get_encoder(self.args.vocab_file, self.args.merge_file, pretokenizer_split)
        return bpe

    def process(self, example):
        args = self.args
        dataset_dir = example['dataset_dir']
        source = infer_source_from_data_dir(dataset_dir)
        if source == 'bigquery':
            standardized_source = 'github'
        else:
            standardized_source = source
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

        attributes = {}
        if 'source' in args.metadata:
            attributes['source'] =  standardized_source

        if 'extension' in args.metadata:
            _, extension = os.path.splitext(example['file_name'])
            if extension is not None and extension.strip():
                if extension == '.ipynb':
                    text, extension = postprocess_ipynb(example['text'])
                else:
                    text = example['text']
                attributes['ext'] = extension

        if 'filename' in args.metadata:
            filename = os.path.basename(example['file_name'])
            if " " in filename:
                filename = f'"{filename}"'
            attributes['filename'] = filename

        if 'dstars' in args.metadata:
            stars = int(example.get('stars', '-1'))
            if (source, setlanguage) in STAR_THRESHOLDS:
                thresholds = STAR_THRESHOLDS[(source, setlanguage)]
                if stars >= thresholds[0]:
                    disc_threshold = 0
                    while disc_threshold < len(thresholds) and thresholds[disc_threshold] <= stars:
                        disc_threshold += 1
                    disc_threshold -= 1
                    assert 0 <= disc_threshold < len(thresholds)

                    attributes['dstars'] = disc_threshold

        if not args.no_redact_pii:
            text = redact_pii(text)

        augmented_file = make_tagged('file', text, attributes=attributes, insert_newlines=True, attribute_move_probability=self.args.attribute_move_probability)
        # encoded = self.encode_text(rank, augmented_file)

        d = {
            'augmented_text': augmented_file,
            # 'augmented_text_tokens': encoded,
            'split': split
        }

        if self.args.maxlen_tokens is not None:
            ids = self.maxlen_bpe.encode(text).ids
            d['over_max_length'] = len(ids) > self.args.maxlen_tokens
        else:
            d['over_max_length'] = False

        return d

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
