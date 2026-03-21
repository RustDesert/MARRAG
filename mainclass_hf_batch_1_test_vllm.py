from openai import OpenAI
import huggingface_hub
import os
import json
import requests
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
from accelerate import Accelerator
from vllm import LLM, SamplingParams
import re

class multi_reflection_rag_hf:
    def __init__(self, planner_client, planner_model):
        self.planner_client = planner_client
        self.planner_model = planner_model
        
        # read hf model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.reflector_model_id = "/volume1/verl/checkpoints/search_r1_like_async_rl_grpo_with_reflection_test/qwen3-4b-it_rm-searchR1-like-sgl_multiturn-with_reflection-2026-03-14-18-37/global_step_462/merged_hf_model"
        
        self.reflector_tokenizer = AutoTokenizer.from_pretrained(self.reflector_model_id)
        # self.reflector_model = AutoModelForCausalLM.from_pretrained(
        #     self.reflector_model_id,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto"
        # )
                
        self.sampling_params_vllm = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        try:
            self.reflector_model_vllm = LLM(
                model=self.reflector_model_id,
                tensor_parallel_size=4,
                dtype="bfloat16",
            )
        except RuntimeError as e:
            print(f"vLLM initialization without GPU memory utilization: {e}")
            print("adding gpu_memory_utilization=0.7 to reduce GPU memory usage")
            self.reflector_model_vllm = LLM(
                model=self.reflector_model_id,
                tensor_parallel_size=4,
                dtype="bfloat16",
                gpu_memory_utilization=0.7,  # Limit memory usage to avoid OOM
            )

    def call_api_planner(self, user_message, system_message=None):
        context = []
        if system_message is not None:
            context.append({"role": "system", "content": system_message})
        if user_message is not None:
            context.append({"role": "user", "content": user_message})
            
        # start to call api
        response = self.planner_client.chat.completions.create(
            model=self.planner_model,
            messages=context,
            response_format = { "type": "json_object" },
        )
        
        return response
    
    def call_api_reflector(self, user_message, system_message=None):
        context = []
        if system_message is not None:
            context.append({"role": "system", "content": system_message})
        if user_message is not None:
            context.append({"role": "user", "content": user_message})
        
        text = self.reflector_tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.reflector_tokenizer([text], return_tensors="pt").to(self.reflector_model.device)
        
        #conduct text completion
        generated_ids = self.reflector_model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        content = self.reflector_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return content
    
    def call_batch_reflector(self, message_list):
        prompt_list = [
            self.reflector_tokenizer.apply_chat_template(
                message, 
                tokenize=False, 
                add_generation_prompt=True,
            ) for message in message_list
        ]
        
        output_list = self.reflector_model_vllm.generate(prompts=prompt_list, sampling_params=self.sampling_params_vllm)
        
        output_list = [output.outputs[0].text for output in output_list]
            
        return output_list

    def search(self, query: str):
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
    
    def generate_sub_questions(self, main_question):
        
        generate_subquestion_system_prompt = """You are an expert reasoning assistant that breaks down a user question into concise, open-ended sub-questions. Rules:\n\nOutput only valid JSON.
        
        JSON must start with "reasoning" (a short paragraph on why these sub-questions are generated and their order).
        
        Then list "1", "2", "3", … as sub-questions..
        
        Keep sub-questions short (≤20 words).
        
        No extra text outside the JSON. 
        
        EXAMPLE JSON OUTPUT:
        {
          "reasoning": "Your reasonings here explaining why these sub-questions are needed.",
          "1": "subquestion 1",
          "2": "subquestion 2",
          "3": "subquestion 3"
        }
        
        """
        
        generate_subquestion_user_prompt = f"Given a question, analyze it, and think, what information do you need to answer this question, and generate related sub-questions for rag search. Do not generate sub-questions which uses the answer given by other sub-questions. \n\n Qustion: {main_question}"
                
        is_valid_json = False
        
        while not is_valid_json:
            try:
                raw_response = self.call_api_planner(user_message=generate_subquestion_user_prompt, system_message=generate_subquestion_system_prompt)
                sub_questions_json = json.loads(raw_response.choices[0].message.content)
                is_valid_json = True
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Retrying...")
                print('-------\nHere is the response failed to read as json (generated_sub_questions)\n-------')
                print(raw_response.choices[0].message.content)
                continue
        
        return raw_response, sub_questions_json
    
    
    def generate_answer_batch(self, sub_question_list, extra_information=None):
        generate_answers_system_prompt = "You are an expert reasoning assistant that generate answer of a question."
        messages = []
        
        def make_prefix(question, retrieved_information):
            prefix = f"""Answer the given question. \
You can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>.\
You must follow the workflow below to answer every question: \
1. Call the search engine to conduct a search for the given question. \
2. Reason step by step inside <reasoning> ... </reasoning> to analyze the search result, and try to  answer the question. \
3. Provide the answer directly inside <answer> and </answer> tags, without detailed illustrations. e.g., <answer>Beijing</answer>. \
4. Reflect on whether the searched information supports the reasoning and generate related reasnonings insde <reflection> and </reflection>. \
5. Output either True or Flase inside <result> ... </result> tags. Output True when the searched information, reasoning, and answer all align; otherwise output False (e.g., if the question is unanswerable or searched information, reasonings and the answer contradict to each other). \n \
Question: {question}\n \
Retrieved information from searching:\n \
{retrieved_information}
"""
            return prefix
        
        retrieves = []
        
        for question in sub_question_list:  
            print("start search")
            search_results = self.search(question)
            print('end search')
            retrieves.append(search_results)
            print(search_results)
            input_message = make_prefix(question, search_results)
        
            if extra_information != None:
                input_message += f'\n\nHere are some extra information to help you answering the question.\n\nExtra information:\n{extra_information}'
            message = [{"role": "system", "content": generate_answers_system_prompt},{"role": "user", "content": input_message}]
            messages.append(message)
        
        raw_response = self.call_batch_reflector(message_list=messages)
        
        def has_tags(text):
            reasoning_opening = text.count("<reasoning>")
            reasoning_closing = text.count("</reasoning>")
            answer_opening = text.count("<answer>")
            answer_closing = text.count("</answer>")
            reflection_opening = text.count("<reflection>") 
            reflection_closing =text.count("</reflection>") 
            result_opening = text.count("<result>") 
            result_closing = text.count("</result>") 
            
            if reasoning_opening >= 1 and reasoning_closing >= 1 and answer_opening >= 1 and answer_closing >= 1 and reflection_opening >= 1 and reflection_closing >= 1 and result_opening >= 1 and result_closing >= 1:
                return True
            else:   
                return False
        
        
        def get_reasoning(text):
            pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_reflection(text):
            pattern = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_answer(text):
            pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_reflection_result(text):
            pattern = re.compile(r"<result>(.*?)</result>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
        
        json_resturn_list = []
        
        for i in range(len(raw_response)):
            full_answer_format = {}
            full_answer_format['question'] = sub_question_list[i]
            full_answer_format['information'] = retrieves[i]
            full_answer_format['reasoning'] = get_reasoning(raw_response[i])
            full_answer_format['answer'] = get_answer(raw_response[i])
            full_answer_format['reflection'] = get_reflection(raw_response[i])
            full_answer_format['reflection_result'] = get_reflection_result(raw_response[i])
            
            answer_json = json.loads(json.dumps(full_answer_format))
            json_resturn_list.append(answer_json)

        return raw_response, json_resturn_list
    
    def generate_answer(self, sub_question, extra_information=None):
        
        search_results = self.search(sub_question)
        print(search_results)

        generate_answers_system_prompt = "You are an expert reasoning assistant that generate answer of a question."
        input_message = f"""Answer the given question. \
You can call a search engine by <tool_call> query </tool_call> and it will return the top searched results between <tool_response> and </tool_response>.\
You must follow the workflow below to answer every question: \
1. Call the search engine to conduct a search for the given question. \
2. Reason step by step inside <reasoning> ... </reasoning> to analyze the search result, and try to  answer the question. \
3. Provide the answer directly inside <answer> and </answer> tags, without detailed illustrations. e.g., <answer>Beijing</answer>. \
4. Reflect on whether the searched information supports the reasoning and generate related reasnonings insde <reflection> and </reflection>. \
5. Output either True or Flase inside <result> ... </result> tags. Output True when the searched information, reasoning, and answer all align; otherwise output False (e.g., if the question is unanswerable or searched information, reasonings and the answer contradict to each other). \n \
Question: {sub_question}\n \
Retrieved information from searching:\n \
{search_results}
"""
        
        if extra_information != None:
            input_message += f'\n\nHere are some extra information to help you answering the question.\n\nExtra information:\n{extra_information}'
            
        raw_response = self.call_api_reflector(user_message=input_message, system_message=generate_answers_system_prompt)

        # answer = json.loads(raw_response.choices[0].message.content)
        # new_answer = "{"
        # new_answer += f'"question": "{sub_question}", "thinking": "{answer["thinking"]}", "answer": "{answer["answer"]}"'
        # new_answer += "}"
        # new_answer = json.loads(new_answer)
        
        def get_reasoning(text):
            pattern = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_reflection(text):
            pattern = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_answer(text):
            pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
            
        def get_reflection_result(text):
            pattern = re.compile(r"<result>(.*?)</result>", re.DOTALL)
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
            else:
                return None
        
        full_answer_format = {}
        full_answer_format['question'] = sub_question
        full_answer_format['information'] = search_results
        full_answer_format['reasoning'] = get_reasoning(raw_response)
        full_answer_format['answer'] = get_answer(raw_response)
        full_answer_format['reflection'] = get_reflection(raw_response)
        full_answer_format['reflection_result'] = get_reflection_result(raw_response)
        
        answer_json = json.loads(json.dumps(full_answer_format))
        return raw_response, answer_json
    
    
    def rewrite_query(self, qa_pair_for_fix, support_info):
        rewrite_query_system_message = """You are an expert reasoning assistant. Your task is to reconstruct the given question with the support information. Think carefully. Output only valid JSON.
        
        EXAMPLE JSON OUTPUT:
        {
            "thinking": "Your reasonings here explaining why you rewrote the question this way.",
            "new_question": "The rewrited question here."
        }
        """
        
        rewrite_query_input_message = f"""You will be provided with a question which has not been answered correctly or unable to be answered. Think step by step, and rewrite the question to make it more answerable. You will be provided with supportive information to help you rewrite the original question. Generate your reasonings and the rewrited question in valid JSON. Generate your reasonings in the JSON with the key "thinking" and generate the question you rewrited with the key "new_question" 
        
        Question for rewrite:
        {qa_pair_for_fix['question']}
        
        Original Answer:
        {qa_pair_for_fix['answer']}
        
        Problems of the original answering:
        {qa_pair_for_fix['reflection']}
        
        support information:
        {support_info}
        """
        
        print(rewrite_query_input_message)
        
        is_valid_json = False
        
        while not is_valid_json:
            try:
                raw_response = self.call_api_planner(system_message=rewrite_query_system_message, user_message=rewrite_query_input_message)
                response = raw_response.choices[0].message.content
                response_json = json.loads(response)
                is_valid_json = True
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Retrying...")
                print('-------\nHere is the response failed to read as json (rewrite query)\n-------')
                print(response)
                continue
            
            
        return raw_response, response_json
    
    def generate_new_sub_questions(self, main_question, supporting_information):
        
        generate_subquestion_system_prompt = """You are an expert reasoning assistant that breaks down a user question into concise, open-ended sub-questions. Rules:\n\nOutput only valid JSON.
        
        JSON must start with "reasoning" (a short paragraph on why these sub-questions are generated and their order).
        
        Then list "1", "2", "3", … as sub-questions..
        
        Keep sub-questions short (≤20 words).
        
        No extra text outside the JSON. 
        
        EXAMPLE JSON OUTPUT:
        {
          "reasoning": "Your reasonings here explaining why these sub-questions are needed.",
          "1": "subquestion 1",
          "2": "subquestion 2",
          "3": "subquestion 3"
        }
        
        """
        
        generate_subquestion_user_prompt = f"""Given a question and supporting information, analyze it, and think, what information do you need to answer this question, and generate related sub-questions for rag search. Do not generate sub-questions which uses the answer given by other sub-questions. Generate at least 3 sub-questions. Do not generate any sub-questions which existed in the supporting information.\n\nQuestion:\n{main_question}\n\nSupporting information:\n{supporting_information}"""
        
        is_valid_json = False
        
        while not is_valid_json:
            try:
                raw_response = self.call_api_planner(user_message=generate_subquestion_user_prompt, system_message=generate_subquestion_system_prompt)
                new_sub_questions_json = json.loads(raw_response.choices[0].message.content)
                is_valid_json = True
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Retrying...")
                print('-------\nHere is the response failed to read as json (generate new sub questions)\n-------')
                print(raw_response.choices[0].message.content)
                continue
    
        
        return raw_response, new_sub_questions_json
    
    def final_answer_analyze(self, question, supporting_information):
        system_prompt ="""You are an expert reasoning assistant. Your task is to generate the final answer of a given query. Think carefully. Output only valid JSON.
        
        EXAMPLE JSON OUTPUT:
        {
            "thinking": "Your reasonings here explaining how you got the final answer.",
            "final_answer": "Final Answer"
        }
        """
        user_input = f"""With a given question and supporting information, think step by step, and generate both your reasoning and the final answer in valid JSON. Generate your reasonings in the key "thinking" and your final answer in the key "final_answer". Do not add detailed illustration or explanation to the answer, generate all your reasonings and explanation to the "thinking" part of your output. For example, if the final answer is "Beijing", just output "Beijing" as the final answer, i.e., "final_answer": "Beijing".
        
        Question:
        {question}
        
        Supporting information:
        {supporting_information}
        """
        
        is_valid_json = False
        
        while not is_valid_json:
            try:
                raw_response = self.call_api_planner(system_message=system_prompt, user_message=user_input)
                response = raw_response.choices[0].message.content
                response_json = json.loads(response)
                is_valid_json = True
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Retrying...")
                print('-------\nHere is the response failed to read as json (final answer analysis)\n-------')
                print(response)
                continue
        
        final_answer = response_json['final_answer']
        
        return raw_response, response_json, final_answer
    
    def execute(self, input_question, question_id):
        print('------- start reasoning -------')
        print(f'Question ID: {question_id}')
        print(input_question)
        subquestion_response, subquestion_json = self.generate_sub_questions(main_question=input_question)
        print(subquestion_json)
        
        subquestion_list = []
        
        for x in subquestion_json:
            if x == "reasoning":
                continue
            else:
                subquestion_list.append(subquestion_json[x])
        print(subquestion_list)
        
        answered_qa_list = []

        qa_raw_answer_list, qa_answer_list = self.generate_answer_batch(sub_question_list=subquestion_list, extra_information=None)
        answered_qa_list.extend(qa_answer_list)

        reflected_true_list = []
        reflected_false_list = []
        
        for x in answered_qa_list:
            if x['reflection_result'] == 'True':
                reflected_true_list.append(x)
            elif x['reflection_result'] == 'False':
                reflected_false_list.append(x)
        print(f"reflected true: \n{reflected_true_list}\n\n")
        print(f"reflected false: \n{reflected_false_list}")
        
        loop_time = 3
        
        for i in range(loop_time):
            print(f"This is loop {i}")
            new_answered_qa_list = []
            new_questions = []
            
            supporting_information_list = []
            
            if reflected_false_list:
                for x in reflected_true_list:
                    information_format = {}
                    information_format['Question'] = x['question']
                    information_format['Reasoning'] = x['reasoning']
                    information_format['Answer'] = x['answer']
                    information_format['Reflection'] = x['reflection']
                    information_format['Reflection Result'] = x['reflection_result']
                    information_json = json.loads(json.dumps(information_format))
                    supporting_information_list.append(information_json)
                
                new_subquestion_response, new_subquestion_json = self.generate_new_sub_questions(main_question=input_question, supporting_information=supporting_information_list)
                
                print(new_subquestion_json)
                for x in new_subquestion_json:
                    if x == "reasoning":
                        continue
                    else:
                        if new_subquestion_json[x] in [info['Question'] for info in supporting_information_list]:
                            continue
                        else:
                            new_questions.append(new_subquestion_json[x])
                print(new_questions)
                
                reflected_false_list = []
                
                qa_raw_answrs, qa_answers = self.generate_answer_batch(sub_question_list=new_questions, extra_information=supporting_information_list)
                new_answered_qa_list.extend(qa_answers)
                
                print(new_answered_qa_list)
                
                for x in answered_qa_list:
                    if x['reflection_result'] == 'True':
                        reflected_true_list.append(x)
                    elif x['reflection_result'] == 'False':
                        reflected_false_list.append(x)
        
        print(f"reflected true: \n{reflected_true_list}\n\n")
        print(f"reflected false: \n{reflected_false_list}")
        
        supporting_information_list = []
        
        for x in reflected_true_list:
            information_format = {}
            information_format['Question'] = x['question']
            information_format['Reasoning'] = x['reasoning']
            information_format['Answer'] = x['answer']
            information_format['Reflection'] = x['reflection']
            information_format['Reflection Result'] = x['reflection_result']
            information_json = json.loads(json.dumps(information_format))
            supporting_information_list.append(information_json)
        
        raw_response, response_json, final_answer = self.final_answer_analyze(question=input_question, supporting_information=reflected_true_list)
        
        print(response_json)
        print(final_answer)
        print('------- end reasoning -------')
        
        json_format = {}
        json_format['question_id'] = question_id
        json_format['input_question'] = input_question
        json_format['reflected_true_list'] = reflected_true_list
        json_format['reflected_false_list'] = reflected_false_list
        json_format['final_answer_analysis'] = response_json
        json_format['final_answer'] = final_answer
        final_output_json = json.loads(json.dumps(json_format))
        
        return final_output_json


def get_last_id(filepath):
    last_line = None
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            last_line = json.loads(f.readlines()[-1].strip())
        except IndexError:
            print("The file is empty or contains only blank lines.")
            last_line = None
        
    if last_line is None:
        raise ValueError("The file is empty or contains only blank lines.")
    
    if "id" not in last_line:
        raise ValueError("The last JSON object does not contain an 'id' field.")
    
    return str(last_line["id"])


if __name__ == "__main__":
    import datasets
    import datetime
    
    huggingface_hub.login(hf_token)

    # deepseek client
    deepseek_client = OpenAI(api_key=ds_token, base_url="https://api.deepseek.com") 
    # models: deepseek-chat, deepseek-reasoner
    
    
    # ----------------------------------------------------------------------
    
    date_now = datetime.datetime.now()
    
    output_path = f"/volume1/multi-agent-rag-reflection/dataset_test_output/"
    os.makedirs(output_path, exist_ok=True)
    json_output_path = os.path.join(output_path, f'hotpotqa-deepseek-chat-test-vllm-2026-03-19-00-25.jsonl')
    
    
    file_exist = os.path.isfile(json_output_path)
    
    if file_exist:
        print(f"{json_output_path} already exists. New results will be appended to the existing file.")
        last_id_str = get_last_id(json_output_path)
        last_id_int = int(last_id_str.split("_")[1])
        print(f"The last question ID in the existing file is: {last_id_str}, in int is: {last_id_int}.")
    else:
        print(f"{json_output_path} does not exist. A new file will be created.")
        open(json_output_path, 'a', encoding='utf-8').close()

    runclass = multi_reflection_rag_hf(
        planner_client=deepseek_client,
        planner_model="deepseek-chat",
    )
    
        
    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa")
    test_dataset = dataset['dev']
    data_size = len(test_dataset)
    
    next_first_id = 0
    
    if file_exist:
        next_first_id = last_id_int +1
    else:
        next_first_id = 0
        
    if next_first_id > data_size:
        print(f"All questions in the dataset have been processed. The last processed question ID is {last_id_str}.")
    else:
        for idx in range(next_first_id, data_size):
            question_id = test_dataset[idx]["id"]
            input_question = test_dataset[idx]["question"]
            golden_answers = test_dataset[idx]["golden_answers"]
            
            outputed_json = runclass.execute(input_question=input_question, question_id=question_id)
            
            final_output_json = {}
            final_output_json['id'] = question_id
            final_output_json['golden_answers'] = golden_answers
            final_output_json['output_answer'] = outputed_json['final_answer']
            final_output_json['output'] = outputed_json
            
            final_output_json = json.loads(json.dumps(final_output_json))
            
            with open(json_output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(final_output_json, ensure_ascii=False) + '\n')
            
            print(f"Finished question {question_id}")
            
        print("--------------\nAll questions finished!\n--------------")