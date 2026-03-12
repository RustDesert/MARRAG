from openai import OpenAI
import transformers
import huggingface_hub
import datasets
import requests
import os
import json
import re
import string

huggingface_hub.login(token=hf_token)



# Qwen3-8b huggingface
qwen3_modelname = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = transformers.AutoTokenizer.from_pretrained(qwen3_modelname)
model = transformers.AutoModelForCausalLM.from_pretrained(
    qwen3_modelname,
    torch_dtype="auto",
    device_map="auto"
)


#dataset

data_source = 'nq'

dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
train_dataset = dataset['train']

output_dir = f"/volume1/multi-agent-rag-reflection/datasets/{data_source}"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, f'{data_source}_nq_reflect.jsonl')

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
    generate_answers_system_prompt = "You are an expert reasoning assistant. Given a question and a series of retrieved information, generate a step-by-step reasoning, answer the question, and perform self-reflection. You may use the retrieved information to help you answering the question."
    
    input_message = f"""Answer the given question. You will be provided with both the question and some retrieved information which could help you answering the question.\
    First reason step-by-step, and wrap the thought process inside <reasoning> and </reasoning> tags.\
    After reasoning, generate your final answer inside <answer> and </answer> tags without detailed illustration.\
    Do not explain anything in <answer></answer> tags, generate your answer directly, e.g., <answer>Beijing</answer>.\
    If you don't know the answer, or the quetion is unanswerable, write \"I don't know\" or \"The question is unanswerable\" inside <answer> and </answer> tags.\
    After you generated your answer, generate reflections on your previous reasonings and the answer you have provided inside <reflection> and <reflection> tags. \
    You should generate reflection from following aspects:\
    - Does the answer directly address the question?\
    - Is the retrieved information relevant to the question?\
    - Does the retrieved information support the answer?\
    - Do you think the answer factually correct and aligned with the ground truth?\
    After reflection, based on the reflection you have generated, do you think the answer you have generated successfully answered the question? Generate either True or False inside <result> and </result> tags. \
    Generate True when you think the answer is correct and is correctly supported by the retrieved information. \
    Generate false in all other cases, e.g., the answer is wrong; the question is unanswerable, the answerer don't know the answer, etc. \
    Do not explain anything in <result> and </result> tags, generate your decision directly. E.g., <result>True</result> or <result>False</result>\
    \
    Strictly follow the format, do not add anything extra.\
    \
    --- Input ---\
    Question:\
    {query}\
    \
    Retrieved information from searching:\
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

train_len = len(train_dataset)

output_list = []



for i in range(100):
    id = train_dataset[i]['id']
    question = train_dataset[i]['question']
    golden_answers = train_dataset[i]['golden_answers']
    
    search_results = search(question)
    
    # search_results = "Doc1: abcdefg"
    message = answer_prompt_format(query=question, retrieved_information=search_results)
    
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_input = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_answer_ids = model.generate(
        **answer_input,
        max_new_tokens=16384
    )
    
    answer_output_ids = generated_answer_ids[0][len(answer_input.input_ids[0]):].tolist() 

    answer_content = tokenizer.decode(answer_output_ids, skip_special_tokens=True)
    
    # full message
    
    full_message = answer_content

    extract_answer = extract_solution(full_message)
    extract_result = extract_result_blocks(full_message)
    
    message.append({"user":"assistant", "content":full_message})

    return_message={"id": id, "messages": message, "golden_answer": golden_answers, "answer": extract_answer, "result": extract_result}

    output_list.append(return_message)

    with open(file_path, 'a', encoding='UTF-8') as f:
        f.write(json.dumps(return_message, ensure_ascii=False)+"\n")
    
# with open(file_path, 'w', encoding='UTF-8') as f:
#     for x in output_list:
#         f.write(json.dumps(x, ensure_ascii=False)+"\n")
    
    
    
