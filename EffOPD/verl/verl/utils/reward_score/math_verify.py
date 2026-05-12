# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


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


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    # verify_func = math_metric(
    #     gold_extraction_target=(LatexExtractionConfig(),),
    #     pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    # )
    # ret_score = 0.0

    # # Wrap the ground truth in \boxed{} format for verification
    # ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    # try:
    #     ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    # except Exception:
    #     pass
    # except TimeoutException:
    #     ret_score = timeout_score

    # return ret_score

    result = False
    answer = remove_boxed(last_boxed_only_string(model_output))
    if answer is None:
        return 0.0

    try:
        if len(answer) > 300:
            answer = answer[:300]
        result = verify(parse("\\boxed{" + ground_truth + "}"), parse("\\boxed{" + answer + "}"))
    except Exception:
        pass

    if result:
        return 1.0
    else:
        return 0.0

