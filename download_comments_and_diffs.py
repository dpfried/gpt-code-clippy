import os
import csv
import random
import shutil
import json
from collections import Counter
from multiprocessing import JoinableQueue
from threading import Thread
from tqdm import tqdm
import argparse
import subprocess

from github_utils.utils import GitRepo
from github_utils.comments import aggregate_comments

class Worker(Thread):
    def __init__(self, args, index, input_queue, output_queue,
                 progress_bar=None, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.index = index
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.progress_bar = progress_bar

    def run(self):
        #print(f"worker {self.index} starting")
        while True:
            input = self.input_queue.get()
            if input is None:
                self.output_queue.put(None)
                self.input_queue.task_done()
                break

            if self.progress_bar is not None and input is not None:
                with self.progress_bar.get_lock():
                    self.progress_bar.set_postfix({'name': input['name']})

            result = self.compute_result(input)

            if self.progress_bar is not None and input is not None:
                with self.progress_bar.get_lock():
                    self.progress_bar.update(1)

            self.output_queue.put(result)
            self.input_queue.task_done()
        # print(f"worker {self.index} ending")
    
    def compute_result(self, input):
        raise NotImplementedError()

class CommentWorker(Worker):
    def compute_result(self, input):
        try:
            owner, repo = input['name'].split('/')
            comment_data = aggregate_comments(owner, repo, approximate_max_comments=self.args.approximate_max_comments)
            input.update(comment_data)
        except Exception as e:
            print(e)
            import traceback
            traceback.print_tb(e.__traceback__, limit=3)
        return input

class FileWorker(Worker):

    def compute_result(self, repo_data):
        args = self.args
        assert args.scratch_dir is not None

        # make output dirs
        tmp_dir = os.path.join(args.scratch_dir, '.tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        repodir = None

        # Dict[commit_id, [List[path: str]]]
        commits_and_paths = repo_data['commits_and_paths']

        def remove_dir():
            if repodir is not None:
                try:
                    shutil.rmtree(repodir, ignore_errors=True)
                except Exception as e:
                    print(e)

        try:
            name = repo_data['name']
            base_url = f'https://github.com/{name}'

            username, projectname = name.split("/")
            rootfolder = os.path.join(tmp_dir, username)
            repodir = os.path.join(rootfolder, projectname)

            command = f'GIT_TERMINAL_PROMPT=0 git clone {base_url} {projectname}'

            os.makedirs(rootfolder, exist_ok=True)
            p = subprocess.Popen(
                command,
                shell=True,
                cwd=rootfolder,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            try:
                p.wait(args.clone_timeout)
            except subprocess.TimeoutExpired:
                print(f'download for {name} timed out ')
                p.kill()

            git_repo = GitRepo(repodir)

            commit_info = []

            bad_commits = set()
            file_data = []

            for commit_and_paths in commits_and_paths:
                commit_id = commit_and_paths['commit_id']
                paths = commit_and_paths['paths']
                try:
                    for path in paths:
                        # dict with keys {'path', 'parent_content', 'parent_commit', 'child_content', 'child_commit'}
                        file_content = git_repo.get_file_at_commit(commit_id, path)
                        file_datum = {
                            'commit_id': commit_id,
                            'path': path,
                            'content': file_content
                        }
                        file_data.append(file_datum)
                    commit_info.append(git_repo.get_commit_info(commit_id, paths))
                except Exception as e:
                    print(e)
                    bad_commits.add(commit_id)
            
            repo_data['file_data'] = file_data
            repo_data['commit_info'] = commit_info
            repo_data['missing_commits'] = list(sorted(bad_commits))

            # if len(bad_commits) > 0:
            #     print(f"{len(bad_commits)} / {len(commits_and_paths)} commits for repo {name} not found")

            out_dir = os.path.join(args.output_dir, "data", username)
            os.makedirs(out_dir, exist_ok=True)

            import zstandard as zstd
            comp = zstd.ZstdCompressor(level=3, threads=4)
            with open(os.path.join(out_dir, f"{projectname}.json.zstd"), 'wb') as f:
                writer = comp.stream_writer(f)
                writer.write(json.dumps(repo_data, indent=4, sort_keys=True).encode('utf-8') + b'\n')
                writer.flush()
                f.flush()

        except Exception as e:
            if not args.abort_on_errors:
                print(e)
                print(f"error dir: {repodir}")
                import traceback
                traceback.print_tb(e.__traceback__, limit=3)
                remove_dir()
                return (name, False, e)
            else:
                remove_dir()
                raise e
        remove_dir()
        return (name, True, None)


class FileDiffWorker(FileWorker):
    def get_file_datum(commit_id, path):
        return git_repo.get_file_before_and_after_commit(commit_id, path)


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('--scratch_dir', default="/scratch/dpf/code-crawl-scratch")
    parser.add_argument('--input_csvs', nargs="+", required=True)
    parser.add_argument('--num_threads', help='number of processes', default=10, type=int)
    parser.add_argument('--clone_timeout', help='timeout for git clone command in seconds',
                        default=480,
                        type=int)
    parser.add_argument('--processing_timeout', help='timeout for processing repo in seconds',
                        default=150,
                        type=int)
    parser.add_argument('--approximate_max_comments', type=int)
    parser.add_argument('-v', '--verbose', help='if flag is present, print errors', action='store_true')
    parser.add_argument('--abort_on_errors', action='store_true')
    parser.add_argument('--retry_failures', action='store_true')
    return parser.parse_args()

class Pipe:
    def __init__(self, args, WorkerClass, num_workers, in_queue=None, out_queue=None, **worker_kwargs):
        self.args = args
        if in_queue is None:
            in_queue = JoinableQueue()
        if out_queue is None:
            out_queue = JoinableQueue()
        self.in_queue, self.out_queue = in_queue, out_queue
        self.workers = []
        self.num_workers = num_workers
        self.WorkerClass = WorkerClass
        self.worker_kwargs = worker_kwargs

    def start(self, pbar_total=None, pbar_name='', pbar_position=None):
        progress_bar = tqdm(total=pbar_total, ncols=120, desc=f"{pbar_name} worker", position=pbar_position)
        for i in range(self.num_workers):
            worker = self.WorkerClass(self.args, i, self.in_queue, self.out_queue, progress_bar=progress_bar, **self.worker_kwargs)
            worker.start()
            self.workers.append(worker)


if __name__ == '__main__':

    args = process_args()  # parse args
    verbose = args.verbose

    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.output_dir

    if not os.path.isdir(args.output_dir):
        raise Exception("output directory {args.output_dir} does not exist; exiting")

    already_processed_filename = os.path.join(args.output_dir, "repos_processed.txt")
    failure_filename = os.path.join(args.output_dir, "repos_failures.txt")
    if os.path.exists(already_processed_filename):
        with open(already_processed_filename, 'r') as f:
            already_processed = set(l.strip() for l in f.readlines())
    else:
        already_processed = {}

    if os.path.exists(failure_filename):
        with open(failure_filename, 'r') as f:
            failures = set(l.strip() for l in f.readlines())
    else:
        failures = {}

    all_repo_data = []

    to_process = set()

    for input_csv in args.input_csvs:
        with open(input_csv, 'r') as f:
            csv_reader = csv.DictReader(f)
            this_repo_data = list(csv_reader)

        # possibly filter out repos from the file in --already_scraped_input
        repo_data_filtered = []

        skipped = Counter()

        for t in this_repo_data:

            assert 'name' in t, f"csv keys {t.keys()} do not include 'name'"

            if t['name'] in already_processed: 
                skipped['processed'] += 1
                continue
            if ((not args.retry_failures) and t['name'] in failures):
                skipped['failed'] += 1
                continue
            if t['name'] in to_process:
                skipped['duplicate'] += 1
                continue

            to_process.add(t['name'])
            repo_data_filtered.append(t)
        print(f"{input_csv}:\tskipping {sum(skipped.values())} repos \t {skipped.most_common()}")
        all_repo_data.extend(repo_data_filtered)

    print(f"{len(all_repo_data)} repos to process")
    already_processed_file = open(already_processed_filename, 'a', buffering=1)
    failure_file = open(failure_filename, 'a', buffering=1)

    print(args.output_dir)
    output_dir = args.output_dir

    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    all_repo_data.sort(key=lambda t: t['name'])
    random.seed(1)
    random.shuffle(all_repo_data)


    os.makedirs(scratch_dir, exist_ok=True)
    os.chdir(scratch_dir)

    comment_pipe = Pipe(args, CommentWorker, args.num_threads)
    file_pipe = Pipe(args, FileDiffWorker, args.num_threads, in_queue=comment_pipe.out_queue)

    num_jobs = len(all_repo_data)

    for repo_data in all_repo_data:
        comment_pipe.in_queue.put(repo_data)
    for _ in range(args.num_threads):
        comment_pipe.in_queue.put(None)

    comment_pipe.start(pbar_name="comments", pbar_position=0, pbar_total=num_jobs)
    file_pipe.start(pbar_name="files", pbar_position=1, pbar_total=num_jobs)

    success_count = 0
    total_count = 0
    while num_jobs > 0:
        r = file_pipe.out_queue.get()
        file_pipe.out_queue.task_done()
        if r is None:
            #print(f"running count: {running_count}")
            continue
        num_jobs -= 1
        # progress_bar.update(1)
        repo_name, was_success, error = r

        if was_success:
            success_count += 1
            already_processed_file.write(repo_name+"\n")
            already_processed_file.flush()
        else:
            failure_file.write(repo_name+"\n")
            failure_file.flush()
        total_count += 1
        
        if total_count % 10 == 0:
            print(f"successes: {success_count} / {total_count} ({success_count/total_count*100:.2f}%)")

    already_processed_file.close()
    failure_file.close()
