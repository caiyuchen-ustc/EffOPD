import argparse
import os
import torch
import json
from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re
from math_verify import parse, verify
import copy
from collections import Counter
import random


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None
    
def apply_chat_template(toker, messages, chat_template=None, enable_thinking=False):
    if chat_template is None:
        input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
    else:
        input_prompt = chat_template.format(prompt=messages[0]["content"])
    return input_prompt


def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--begin_idx", type=int, default=-1)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    

    toker = AutoTokenizer.from_pretrained(args.model_path)
    args.model_name = os.path.basename(args.model_path)

    llm = LLM(
        model=args.model_path, tokenizer=args.model_path,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=args.max_num_seqs,
    )

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_tokens, n=args.n, seed=args.seed)


    with open(args.input_file, "r", encoding="utf-8") as file:
        input_data = [json.loads(line) for line in file]

    if args.begin_idx >= 0 and args.end_idx >= 0:
        input_data = input_data[args.begin_idx: args.end_idx]


    prompts = []
    for item in input_data:
        problem = item["problem"]
        answer = str(item["answer"])
        item["answer"] = answer
        problem = problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompts.append(problem)

    chat_template = None
    
    prompt_token_ids = [apply_chat_template(toker, [{"role": "user", "content": prompt}], chat_template=chat_template, enable_thinking=args.enable_thinking)
                            for prompt in prompts]
    
    generations = llm.generate(prompt_token_ids, sampling_params=sampling_params)


    res_data = []
    for i in range(len(input_data)):
        d = copy.deepcopy(input_data[i])
        # For each input, collect all responses and boxed answers
        responses = []
        boxed_answers = []
        acc_list = []
        # There are args.n generations per input
        for j in range(len(generations[i].outputs)):
            response = generations[i].outputs[j].text.strip()
            responses.append(response)
            boxed_answer = remove_boxed(last_boxed_only_string(response))
            boxed_answers.append(boxed_answer)
            # Compare boxed_answer with d["answer"]
            if boxed_answer is None:
                acc = False
            else:
                acc = verify(parse("\\boxed{" + d["answer"] + "}"), parse("\\boxed{" + boxed_answer + "}"))
            acc_list.append(acc)
        d["pred_answers"] = boxed_answers
        d["responses"] = responses
        d["acc_list"] = acc_list
        d["model"] = args.model_name
        res_data.append(d)

    total_preds = 0
    correct_preds = 0
    pass_at_k = 0
    avg_length = 0
    for d in res_data:
        accs = d.get("acc_list", [])
        total_preds += len(accs)
        correct_preds += sum(1 for acc in accs if acc)
        if any(acc for acc in accs):
            pass_at_k += 1
        responses = d.get("responses", [])
        for response in responses:
            length = len(toker.encode(response, add_special_tokens=False))
            avg_length += length / len(responses)

    accuracy = correct_preds / total_preds if total_preds > 0 else 0.0
    pass_at_k  = pass_at_k / len(res_data) if len(res_data) > 0 else 0.0
    avg_length = avg_length / len(res_data) if len(res_data) > 0 else 0.0
    print(f"dataset: {args.input_file}")
    print(f"Total predictions: {total_preds}")
    print(f"Accurate predictions: {correct_preds}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"passs@k: {pass_at_k:.4f}")
    print(f"avg_length: {avg_length:.4f}")
    with open(args.output_file, "w", encoding="utf-8") as file:
        for d in res_data:
            file.write(json.dumps(d) + "\n")


if __name__ == '__main__':
    main()
