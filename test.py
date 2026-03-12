from openai import OpenAI
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
import datasets
import requests
import os
import torch
import json

huggingface_hub.login(token=hf_token)

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


#dataset

data_source = 'nq'

dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
train_dataset = dataset['train']

output_dir = f"/volume1/multi-agent-rag-reflection/datasets/{data_source}"
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, f'{data_source}_train_reflect.jsonl')



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# prepare the model input
prompt = f"""Answer the given question. You will be provided with both the question and some retrieved information which could help you answering the question.\
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
Generate True when you think the answer is correct and is correctly supported by the retrieved information. Generate false in all other cases, e.g., the answer is wrong; the question is unanswerable, the answerer don't know the answer, etc. Do not explain anything in <result> and </result> tags\
Do not explain anything in <result> and </result> tags, generate your decision directly. E.g., <result>True</result> or <result>False</result>\
\
Strictly follow the format, do not add anything extra.\
\
--- Input ---\
Question:\
where was practical magic supposed to be set\
\
Retrieved information from searching:\
Doc 1(Title: \"Practical Magic\") a risk on love again. Sally sends out a message via a leaf on the wind, and in Arizona, Gary senses her call. He returns to Massachusetts to be with Sally. Halloween is celebrated, and all the Owens women, demonstrating their powers by leaping off their roof and landing safely, are lovingly accepted by the townsfolk. \"\"Practical Magic\"\" was filmed in part on an artificial set in California. Because the film's producers decided the house was a big part of the depiction of the Owens' culture, a house to accurately represent that vision was built on San Juan Island in\
Doc 2(Title: \"Practical Magic (novel)\") Practical Magic (novel) Practical Magic is a 1995 novel by Alice Hoffman. The book was adapted into a 1998 film of the same name. For more than two hundred years, the Owens women have been blamed for everything that has gone wrong in their Massachusetts town. Gillian and Sally have also endured that fate: As children, the sisters were forever outsiders, taunted, talked about, pointed at. Their elderly aunts almost seemed to encourage the whispers of witchery, with their darkened house, their love concoctions and their crowd of black cats. All Gillian and Sally wanted to do was escape. One\
Doc 3(Title: \"Practical Magic\") the state of Washington. While much of the set from California was brought to that location and placed inside the house, it took nearly a year to perfect the image of the house and the interior. The house, actually only a shell with nothing inside, was built only for this filming and was torn down after filming was completed. The small town scenes were filmed on the main street of Coupeville, Washington, a Victorian-era seaside port town located on the south side of Penn's Cove on Whidbey Island. According to Sandra Bullock in the DVD commentary, while filming the scene"""


messages = [
    {"role": "system", "content": "You are an expert reasoning assistant. Given a question and a series of retrieved information, generate a step-by-step reasoning, answer the question, and perform self-reflection. You may use the retrieved information to help you answering the question."},
    {"role": "user", "content":prompt},
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)