import re
import os
import sys
import pprint
import itertools
from os.path import splitext
from collections import Counter, defaultdict, namedtuple
from multiprocessing import Pool, Process, JoinableQueue

from code_clippy_dataset.utils import infer_source_from_data_dir, load_dataset_infer

import tqdm
import humanize
import numpy as np

from datasets import load_dataset, load_from_disk
#from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer

import sentencepiece

from hacky_linguist import COMMON_LANGUAGES, EXTENSION_TO_LANGUAGE

# HF: Whether or not to add an initial space to the input. This allows to treat
# the leading word just as any other word. (GPT2 tokenizer detect beginning of
# words by the preceding space).
ADD_PREFIX_SPACE = False

#LANGUAGE_SOURCES = ['repo_language', 'guesslang', 'filename_extension']
#LANGUAGE_SOURCES = ['repo_language', 'filename_extension', 'linguist']
LANGUAGE_SOURCES = ['linguist']

POSSIBLE_TOKENIZERS = ["gpt2", "codet5", "ours", "sentencepiece", "sentencepiece_rn"]

from train_multicorp_tokenizer import NEWLINE_REP

class Worker(Process):
    def __init__(self, index, input_queue, output_queue, progress_bar=None, tokenizer_names=POSSIBLE_TOKENIZERS, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.progress_bar = progress_bar
        self.tokenizer_names = tokenizer_names[:]
        for tn in tokenizer_names:
            assert tn in POSSIBLE_TOKENIZERS, f"invalid tokenizer {tn}"

    def run(self):
        #print(f"worker {self.index} starting")
        if 'guesslang' in LANGUAGE_SOURCES:
            from guesslang import Guess
            guess = Guess()
        else:
            guess = None
        # guess = None
        tokenizers = {}
        if "gpt2" in self.tokenizer_names:
            #tokenizers["gpt2"] = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=ADD_PREFIX_SPACE)
            tokenizers["gpt2"] = ByteLevelBPETokenizer.from_file(
                "/private/home/halilakin/src/gpt2_bpe/encoder.json",
                "/private/home/halilakin/src/gpt2_bpe/vocab.bpe",
            )
        if "codet5" in self.tokenizer_names:
            codet5_tokenizer_model = ByteLevelBPETokenizer.from_file(
                "../CodeT5/tokenizer/salesforce/codet5-vocab.json",
                "../CodeT5/tokenizer/salesforce/codet5-merges.txt"
            )
            codet5_tokenizer_model.add_special_tokens([
                    "<pad>",
                    "<s>",
                    "</s>",
                    "<unk>",
                    "<mask>"
            ])
            tokenizers["codet5"] = codet5_tokenizer_model
            #tokenizers["codet5"] = PreTrainedTokenizerFast(tokenizer_object=codet5_tokenizer_model)
        if "ours" in self.tokenizer_names:
            our_tokenizer_model = ByteLevelBPETokenizer.from_file(
                "/checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/vocab.json",
                "/checkpoint/dpf/data/tokenizers/github-py-redacted+so_psno-True/merges.txt",
                pretokenizer_split_newlines_only=True,
            )
            tokenizers["ours"] = our_tokenizer_model
        if "sentencepiece" in self.tokenizer_names:
            sp_tokenizer = sentencepiece.SentencePieceProcessor()
            SPLIT_LINES = re.compile(f'.*[\r\n]+')
            sp_tokenizer.Load("tokenizers/github-py+so_spm-rn-False.model")
            def sp_tokenize(text):
                pieces = re.findall(SPLIT_LINES, text)
                if not pieces:
                    pieces = [text]
                return {
                    'input_ids': [token for piece in pieces for token in sp_tokenizer.EncodeAsIds(piece)]
                }
            tokenizers["sentencepiece"] = sp_tokenize
        if "sentencepiece_rn" in self.tokenizer_names:
            sp_tokenizer_rn = sentencepiece.SentencePieceProcessor()
            sp_tokenizer.Load("tokenizers/github-py+so_spm-rn-True.model")
            def sp_tokenize_rn(text):
                text = text.replace("\n", NEWLINE_REP)
                return {
                    'input_ids': sp_tokenizer_rn.EncodeAsIds(text)
                }
            tokenizers["sentencepiece_rn"] = sp_tokenize_rn

        while True:
            x = self.input_queue.get()
            if self.progress_bar:
                with self.progress_bar.get_lock():
                    self.progress_bar.update(1)
            if x is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break
            tabulated = self.tabulate_single(guess, tokenizers, x)
            self.output_queue.put((x, tabulated))
            self.input_queue.task_done()
        print(f"worker {self.index} ending")

    @staticmethod
    def tabulate_single(guess, tokenizers, x):
        d = {}
        for ls in LANGUAGE_SOURCES:
            if ls == 'repo_language':
                found_language = False
                for key in ['repo_language', 'main_language']:
                    if not found_language and key in x:
                        d[ls] = x[key]
                        found_language = True
                assert found_language
            elif ls == 'guesslang':
                d[ls] = guess.language_name(x['text'])
            elif ls == 'filename_extension' or ls == 'linguist':
                fname_toks = splitext(x['file_name'])
                if len(fname_toks) > 1:
                    if ls == 'filename_extension':
                        d[ls] = fname_toks[-1]
                    elif ls == 'linguist':
                        d[ls] = EXTENSION_TO_LANGUAGE[fname_toks[-1]]
                else:
                    d[ls] = None

        d['file_size'] = len(x['text'])

        for tokenizer_name, tokenizer in tokenizers.items():
            tokens = tokenizer.encode(x['text']).ids
            token_count = len(tokens)
            d[f'{tokenizer_name}_token_count'] = token_count
            # d[f'{tokenizer_name}_above_1024'] = 1.0 if token_count > 1024 else 0.0
            # d[f'{tokenizer_name}_above_2048'] = 1.0 if token_count > 2048 else 0.0
            # d[f'{tokenizer_name}_lost_1024'] = max(token_count - 1024, 0)
            # d[f'{tokenizer_name}_lost_2048'] = max(token_count - 2048, 0)
        return d

def foo(train):
    data = []
    for x in train:
        data.append(x)
        if len(data) >= 100:
            break
    return tabulate(data, n_procs=4)

def readable(x, is_size=False):
    if isinstance(x, float):
        return f"{x:.2f}"
    else:
        if is_size:
            return humanize.naturalsize(x)
        else:
            return f"{x:_}"

class WeightedSum:
    def __init__(self):
        self.sum = 0
        self.total_weight = 0

    def add(self, value, weight=1.0):
        self.sum += value
        self.total_weight += weight

    @property
    def mean(self):
        if self.total_weight > 0:
            return float(self.sum) / self.total_weight
        else:
            return 0.0

def agg_token_counts(token_counts, agg_fn=np.sum, human_readable=True, limit=None, is_size=False):
    def inner_display(d):
        d['total'] = sum(d.values())
        if human_readable:
            return {k: readable(v, is_size=is_size) for k, v in Counter(d).most_common(limit)}
        else:
            return d
    return {
        language_source: inner_display({
            language: agg_fn(values)
            for language, values in inner.items()
        })
        for language_source, inner in token_counts.items()
    }

def size_counter_to_human_readable(f_counter, limit=None):
    return {k: humanize.naturalsize(v) for k, v in f_counter.most_common(limit)}

def display_counts(file_counts, tabulated, sum_stats, mean_stats):
    pprint.pprint(file_counts)
    for stat in sum_stats:
        print(f"{stat} sum:")
        pprint.pprint(agg_token_counts(tabulated[stat], lambda ws: ws.sum, is_size='size' in stat))
    for stat in mean_stats:
        print(f"{stat} mean:")
        pprint.pprint(agg_token_counts(tabulated[stat], lambda ws: ws.mean, is_size='size' in stat))

def tabulate(data, tokenizer_names, n_procs=10, max_items=None):
    file_counts = defaultdict(Counter)

    # tokenizer, language source, language
    tabulated = defaultdict(lambda: defaultdict(lambda: defaultdict(WeightedSum)))

    sum_stats = ['file_size']
    mean_stats = ['file_size']

    if tokenizer_names:
        for tokenizer in tokenizer_names:
            sum_stats.append(f'{tokenizer}_token_count')
            mean_stats.append(f'{tokenizer}_token_count')

            # sum_stats.append(f'{tokenizer}_lost_1024')
            # sum_stats.append(f'{tokenizer}_lost_2048')

            # mean_stats.append(f'{tokenizer}_above_1024')
            # mean_stats.append(f'{tokenizer}_above_2048')

    all_stats = list(set(sum_stats) | set(mean_stats))

    in_queue, out_queue = JoinableQueue(), JoinableQueue()

    workers = []

    running_count = 0

    for i in range(n_procs):
        # signal to stop
        worker = Worker(i, in_queue, out_queue, tokenizer_names=tokenizer_names)
        worker.start()
        running_count += 1
        workers.append(worker)

    if max_items is not None:
        data = itertools.islice(data, max_items)
        num_items = max_items
    else:
        num_items = len(data)

    num_jobs = 0
    for x in tqdm.tqdm(data, ncols=80, desc="loading queue"):
        in_queue.put(x)
        num_jobs += 1

    for i in range(n_procs):
        in_queue.put(None)

    file_count = 0
    with tqdm.tqdm(total=num_jobs, ncols=80, desc="processing") as progress_bar:
        while num_jobs > 0:
            r = out_queue.get()
            if r is None:
                running_count -= 1
                #print(f"running count: {running_count}")
                out_queue.task_done()
                continue
            datum, this_tabulated = r
            num_jobs -= 1
            
            for language_source in LANGUAGE_SOURCES:
                language = this_tabulated[language_source]
                file_counts[language_source][language] += 1
                #for tokenizer in ['gpt2', 'codet5', 'ours']:
                for stat in all_stats:
                    tabulated[stat][language_source][language].add(this_tabulated[stat], 1)
            progress_bar.update(1)
            out_queue.task_done()
            file_count += 1
            if file_count % 100000 == 0:
            #if file_count % 100 == 0:
                print(f"total files: {file_count}")
                display_counts(file_counts, tabulated, sum_stats, mean_stats)
                print()

        #[worker.join() for worker in workers]

    # print("calling inqueue.join")
    # in_queue.join()
    # print("calling outqueue.join")
    # out_queue.join()
    print("done")

    return file_counts, tabulated, sum_stats, mean_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    #parser.add_argument("--source", default='github')
    parser.add_argument("--num_items", type=int)
    parser.add_argument("--n_procs", type=int, default=10)
    parser.add_argument("--tokenizer_names", nargs='*', choices=POSSIBLE_TOKENIZERS, default=POSSIBLE_TOKENIZERS)
    args = parser.parse_args()

    print(' '.join(sys.argv))

    data_dir = args.data_dir
    # datasets are hashed based on arguments, which are sensitive to trailing dashes (even though loading is not)
    data_dir = data_dir.rstrip("/")
    print(data_dir)

    dataset = load_dataset_infer(data_dir)

    file_counts, tabulated, sum_stats, mean_stats = tabulate(
        dataset, tokenizer_names=args.tokenizer_names, n_procs=args.n_procs, max_items=args.num_items
    )
    display_counts(file_counts, tabulated, sum_stats, mean_stats)
