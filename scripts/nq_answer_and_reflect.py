from openai import OpenAI
import transformers
import huggingface_hub
import datasets
import requests
import os
import json

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
file_path = os.path.join(output_dir, f'{data_source}_train_reflect.jsonl')

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
    generate_answers_system_prompt = "You are an expert reasoning assistant. Given a question and a series of retrieved information, generate a step-by-step reasoning and answer the question You may use the retrieved information to help you answering the question."
    
    input_message = f"Answer the given question. You will be provided with both the question and some retrieved information which could help you answering the question.\nFirst reason step-by-step, and wrap the thought process inside <reasoning> and </reasoning> tags.\nAfter reasoning, generate your final answer inside <answer> and </answer> tags without detailed illustration.\nDo not explain anything in <answer></answer> tags, generate your answer directly.\nIf you don't know the answer, or the quetion is unanswerable, write \"I don't know\" or \"The question is unanswerable\" inside <answer> and </answer> tags.\nStrictly follow the format, do not add anything extra.\n\n--- Input ---\nQuestion:\n{query}\n\nRetrieved information from searching:\n{retrieved_information}"
    
    message = [{"role": "system","content": generate_answers_system_prompt},{"role": "user","content": input_message}]
    return message

def reflect_prompt_format(query, retrieved_information, answer):
    generate_reflection_prompt_system_prompt = "You are an expert reasoning assistant. Your task is to reflect on the given question and answer. Think carefully."

    input_message = f"You are an expert reasoning assistant. Your task is to reflect on the quality of the given question and answer. \n\nGenerate the reflection from the following aspects:\n- Does the answer directly address the question?\n- Is the retrieved information relevant to the question?\n- Does the retrieved information support the answer?\n- Do you think the answer factually correct and aligned with the ground truth?\n\nGenerate your step-by-step reasonings inside <reflection> and </reflection> tags.\nAfter you finished your reasonings, generate either True or False based on your reasonings which indicate wether the answer correctly answered the question or not inside <reflection_result> and </reflection_result> tags. Generate True when you think the answer is correct and is correctly supported by the retrieved information. Generate false in all other cases, e.g., the answer is wrong; the question is unanswerable, the answerer don't know the answer, etc. Do not explain anything in <reflection_result></reflection_result> tags, generate your decision directly.\nStrictly follow the format, do not add anything extra.\n\n---input---\nQuestion:\n{query}\n\nRetrieved Information:\n{retrieved_information}\n\nAnswer content:\n{answer}"
        
    message = [{"role": "system","content": generate_reflection_prompt_system_prompt},{"role": "user","content": input_message}]
        
    return message

train_len = len(train_dataset)

output_list = []



for i in range(train_len):
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
    
    # Generate reflection
    
    reflect_message = reflect_prompt_format(query=question, retrieved_information=search_results, answer=answer_content)
    
    reflect_text = tokenizer.apply_chat_template(
        reflect_message,
        tokenize=False,
        add_generation_prompt=True,
    )
    reflect_input = tokenizer([reflect_text], return_tensors="pt").to(model.device)
    
    generate_reflect_ids = model.generate(
        **reflect_input,
        max_new_tokens=16384
    )
    
    reflect_output_ids = generate_reflect_ids[0][len(reflect_input.input_ids[0]):].tolist() 
    
    reflect_content = tokenizer.decode(reflect_output_ids, skip_special_tokens=True)
    
    # full message
    
    full_message = answer_content + reflect_content
    
    message.append({"user":"assistant", "content":full_message})

    return_message={"id": id, "messages": message, "golden_answer": golden_answers}

    output_list.append(return_message)

    with open(file_path, 'a', encoding='UTF-8') as f:
        f.write(json.dumps(return_message, ensure_ascii=False)+"\n")
    
# with open(file_path, 'w', encoding='UTF-8') as f:
#     for x in output_list:
#         f.write(json.dumps(x, ensure_ascii=False)+"\n")
    
    
    
