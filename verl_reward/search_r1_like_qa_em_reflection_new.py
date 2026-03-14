# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_bool_result(solution_str):
    boolean_result_pattern = r'<result>(.*?)</result>'
    match = re.finditer(boolean_result_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) < 1:
        return None
    
    return matches[-1].group(1).strip()

def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags

def _has_reflection_block(solution_str):
    """Check if solution has reasoning block with correct spelling."""
    return "<reflection>" in solution_str and "</reflection>" in solution_str

def compute_score(solution_str, ground_truth, method="strict", reflection_format_score=0.3, verification_score=0.3, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    has_reflection = _has_reflection_block(solution_str=solution_str)
    bool_answer = extract_bool_result(solution_str=solution_str)
    bool_result = str(bool_answer).lower()
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
            print(f"Bool answer extracted is: {str(bool_answer)}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")
        
    # R_format: has_reflection and has_bool_result == True
    # R_result

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if has_reflection:
                if bool_result == 'true':
                    return score
                elif bool_result == 'false':
                    return score - verification_score
                else:
                    return 0
            else:
                if bool_result == 'true':
                    return score - reflection_format_score
                elif bool_result == 'false':
                    return score - reflection_format_score - verification_score
                else: return 0
        else:
            if has_reflection:
                if bool_result == 'true':
                    return reflection_format_score 
                elif bool_result == 'false':
                    return reflection_format_score + verification_score
                else: 
                    return 0
            else:
                if bool_result == 'true':
                    return 0
                elif bool_result == 'false':
                    return verification_score
                else: 
                    return 0