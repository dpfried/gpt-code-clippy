import json
# import torch
import pandas as pd

# import apps.eval.reident

# from apps_utils.generate_gpt_codes import generate_prompt
# from apps_utils.test_one_solution import eval_and_save_problems
from datasets import load_dataset, load_metric
from fastcore.script import *
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from pathlib import Path
from tqdm.auto import tqdm
# from metrics.extrinsic_eval import compute_metrics
from subprocess import check_output
from transformers import (
    AutoTokenizer,
    FlaxGPTNeoForCausalLM,
)

bleu = load_metric("sacrebleu")

MAX_TOKENs = 1024
model_name_or_path = "flax-community/gpt-code-clippy-125M-1024-f" # "flax-community/gpt-code-clippy-125M-bs2048-raw" # "EleutherAI/gpt-neo-125M"
branch = "main"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, padding_side="left", pad_token="<|endoftext|>"
)
model = FlaxGPTNeoForCausalLM.from_pretrained(
    model_name_or_path,
    pad_token_id=50256,
    revision=branch
)


def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="jax")  # .to("cuda")
    output_seq = model.generate(input_ids=inputs.input_ids, max_length=MAX_TOKENs, early_stopping=True)
    output = tokenizer.decode(output_seq["sequences"][0])#, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print(output)
    return output


def _eval_concode(path):
    # TODO: format input to model same as App and OpenAI HumanEval datasets are formatted
    data = load_dataset("json", data_files=str(path / "test.json"))["train"]
    predictions = [[]]
    references = []
    for example in data:
        output = generate_text(example["nl"])
        predictions[0].append(output.split(" "))
        references.append(example["code"].split(" "))
    results = compute_metrics(predictions, references)
    print(f"Bleu score for Concode dataset: {results}")


def _eval_apps(path):

    gpt_codes = {}
    apps_ds = load_dataset("../data_processing/apps.py")["test"]
    for idx, example in tqdm(enumerate(apps_ds)):
        answer = generate_text(example["question"])
        gpt_codes[idx] = answer
    with open(path.parent / "all_codes.json", "w") as f:
        json.dump(gpt_codes, f)

    eval_and_save_problems(path, path.parent)
        # prompt = generate_prompt(
        #     Args(),
        #     test_case_path,
        #     prompt_path,
        #     solutions_path,
        #     tokenizer,
        #     starter_path=starter_path,
        # )





    prob_paths = sorted(path.glob("*/"))
    # map prob_paths to strings and save as a json file
    str_paths = [str(p) for p in prob_paths]
    with open(path / "test.json", "w") as f:
        json.dump(str_paths, f)
    for index, prob_path in enumerate(prob_paths[:2]):
        test_case_path = prob_path / "input_output.json"
        prompt_path = prob_path / "question.txt"
        starter_path = prob_path / "starter_code.py"
        solutions_path = prob_path / "solutions.json"
        if not starter_path.exists():
            starter_path = None
        if not test_case_path.exists() or not prompt_path.exists():
            continue
        prompt = generate_prompt(
            Args(),
            test_case_path,
            prompt_path,
            solutions_path,
            tokenizer,
            starter_path=starter_path,
        )
        output = generate_text(prompt)
        print(output)
        # print(output)
        gpt_codes[index] = output
        # print(output)

    with open(path.parent / "all_codes.json", "w") as f:
        json.dump(gpt_codes, f)

    eval_and_save_problems(path, path.parent)

    # execute bash command to run eval script
    # results = check_output(
    #     [
    #         # python3 test_one_solution.py -t /path/to/apps/test --save /path/to/save_dir --print_results
    #         "python",
    #         "./apps_utils/test_one_solution.py",
    #         "-t",
    #         str(path),
    #         "--save",
    #         str(path.parent),
    #         "--print_results",
    #     ]
    # ).decode("utf-8")


#     test_case_path = os.path.join(prob_path, "input_output.json")
#     prompt_path = os.path.join(prob_path, "question.txt")
#     starter_path = os.path.join(prob_path, "starter_code.py")
#     solutions_path = os.path.join(prob_path, "solutions.json")
#  generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None)


def _eval_human_eval(path):
    # problems = read_problems(str(path))
    # num_samples_per_task = 1
    # samples = [
    #     dict(
    #         task_id=task_id,
    #         completion=generate_text(problems[task_id]["prompt"]),
    #     )
    #     for task_id in tqdm(list(problems.keys()))
    #     for _ in range(num_samples_per_task)
    # ]
    # write_jsonl("human_eval.jsonl", samples)
    # execute bash command to run eval script
    results = evaluate_functional_correctness("human_eval.jsonl", [1], 4, 3.0, str(path))
    # results = check_output(
    #     [
    #         "python",
    #         path / "evaluate_functional_correctness.py",
    #         "human_eval.jsonl",
    #     ]
    # ).decode("utf-8")

    print(results)


@call_parse
def main(
    concode_path: Param("Path to the concode data in CodeXGLUE", str),
    apps_path: Param("Path to the the App dataset", str),
    human_eval_path: Param("Path to the human eval dataset", str),
):
    concode_path = Path(concode_path)
    apps_path = Path(apps_path)
    human_eval_path = Path(human_eval_path)
    # _eval_concode(concode_path)
    _eval_human_eval(human_eval_path)
    # _eval_apps(apps_path)
    # dataset = load_dataset("json", data_files=str(concode_path / "test.json"))
    # print(dataset)
    # results = bleu.compute(predictions=predictions, references=references)
    # print(list(results.keys()))
    # print(round(results["score"], 1))


# problems = read_problems()
# print(problems)
# num_samples_per_task = 200
# samples = [
#     dict(
#         task_id=task_id,
#         completion=generate_text(problems[task_id]["prompt"]),
#     )
#     for task_id in problems[:1]
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("human_eval.jsonl", samples)