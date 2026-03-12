from openai import OpenAI
import transformers
import huggingface_hub
import datasets
import requests
import os
import json
import re
import string
#dataset

data_source = 'nq'
split = 'dev'

print("start to retrieve dataset")
datasets = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
target_dataset = datasets[split]
print('dataset retrieved')

print("generate output path")
output_dir = f"/volume1/multi-agent-rag-reflection/datasets/{data_source}"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, f'{data_source}_{split}_prompt_only_with_information.jsonl')

def search(query: str):
    payload = {
            "queries": [query],
            "topk": 3,
            "return_scores": True
        }
    results = requests.post("http://127.0.0.1:8000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])

def answer_prompt_format(query, retrieved_information):
    generate_answers_system_prompt = "You are an expert reasoning assistant. Given a question and a series of retrieved information, generate a step-by-step reasoning, answer the question, and perform self-reflection with step-by-step reasoning. You may use the retrieved information to help you answering the question."
    
    input_message = f"""Answer the given question. You will be provided with both the question and some retrieved information which could help you answering the question.\n\
    First reason step-by-step, and wrap the thought process inside <reasoning> and </reasoning> tags.\n\
    After reasoning, generate your final answer inside <answer> and </answer> tags without detailed illustration.\n\
    Do not explain anything in <answer></answer> tags, generate your answer directly, e.g., <answer>Beijing</answer>.\n\
    If you don't know the answer, or the quetion is unanswerable, write \"I don't know\" or \"The question is unanswerable\" inside <answer> and </answer> tags.\n\
    After you generated your answer, generate reflections on your previous reasonings and the answer you have provided inside <reflection> and <reflection> tags. \n\
    You should generate reflection from following aspects:\n\
    - Does the answer directly address the question?\n\
    - Is the retrieved information relevant to the question?\n\
    - Does the retrieved information support the answer?\n\
    - Do you think the answer factually correct and aligned with the ground truth?\n\
    After reflection, based on the reflection you have generated, do you think the answer you have generated successfully answered the question? Generate either True or False inside <result> and </result> tags. \n\
    Generate True when you think the answer is correct and is correctly supported by the retrieved information. \n\
    Generate false in all other cases, e.g., the answer is wrong; the question is unanswerable, the answerer don't know the answer, etc. \n\
    Do not explain anything in <result> and </result> tags, generate your decision directly. E.g., <result>True</result> or <result>False</result>\n\
    Strictly follow the format, do not add anything extra.\n\
    \n\
    --- Input ---\n\
    Question:\n\
    {query}\n\
    \n\
    Retrieved information:\n\
    {retrieved_information}
    """

    
    message = [{"role": "system","content": generate_answers_system_prompt},{"role": "user","content": input_message}]
    return message


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    return matches[-1].group(1).strip()

def extract_result_blocks(text: str):
    pattern = r"<result>(.*?)</result>"
    match = re.finditer(pattern, text, re.DOTALL)
    matches = list(match)
    return matches[-1].group(1).strip()

train_len = len(target_dataset)

output_list = []


print("start construct")
for i in range(train_len):
    print(i)
    nq_id = target_dataset[i]['id']
    question = target_dataset[i]['question']
    golden_answers = target_dataset[i]['golden_answers']
    
    search_results = search(question)
    
    # search_results = "Doc1: abcdefg"
    message = answer_prompt_format(query=question, retrieved_information=search_results)
    
    # data = {
    #     "id": nq_id,
    #     "question": question,
    #     "golden_answer": golden_answers,
    #     "data_source": data_source,
    #     "prompt": message,
    #     "ability": "fact-reasoning",
    #     "reward_model":{
    #         "style": "rule",
    #         "ground_truth": {"target": golden_answers}
    #     },
    #     "extra_info":{
    #         'split': split,
    #         'index': i
    #     }
    # }
    
    data = {
        "id": nq_id,
        "question": question,
        "golden_answer": golden_answers,
        "data_source": data_source,
        "prompt": message,
        "ability": "fact-reasoning",
        "reward_model":{
            "style": "rule",
            "ground_truth": {"target": golden_answers}
        },
        "extra_info":{
            'split': split,
            'index': i
        }
    }

    with open(file_path, 'a', encoding='UTF-8') as f:
        f.write(json.dumps(data, ensure_ascii=False)+"\n")
    
# with open(file_path, 'w', encoding='UTF-8') as f:
#     for x in output_list:
#         f.write(json.dumps(x, ensure_ascii=False)+"\n")
    
    
    
