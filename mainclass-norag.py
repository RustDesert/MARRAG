from openai import OpenAI
import huggingface_hub
import os
import json

class multi_reflection_rag:
    def __init__(self, planner_client, reflector_client):
        self.planner_client = reflector_client
        self.reflector_client = reflector_client
        
        self.planner_model = "deepseek-chat"
        self.reflector_model = "deepseek-chat"
                
        self.question_list = []
        self.information_list = []
        self.neg_reflection_qa_list = []

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
            stream = False,
        )
        
        return response
    
    def call_api_reflector(self, user_message, system_message=None):
        context = []
        if system_message is not None:
            context.append({"role": "system", "content": system_message})
        if user_message is not None:
            context.append({"role": "user", "content": user_message})
            
        # start to call api
        response = self.reflector_client.chat.completions.create(
            model=self.planner_model,
            messages=context,
            response_format = { "type": "json_object" },
            stream = False,
        )
        
        return response
    
    def generate_sub_questions(self, main_question):
        
        generate_subquestion_system_prompt = f"You are an expert reasoning assistant that breaks down a user question into concise, open-ended sub-questions. Rules:\n\nOutput only valid JSON.\n\nJSON must start with \"reasoning\" (a short paragraph on why these sub-questions are generated and their order).\n\nThen list \"1\", \"2\", \"3\", … as sub-questions..\n\nKeep sub-questions short (≤20 words).\n\nNo extra text outside the JSON."
        
        generate_subquestion_user_prompt = f"Given a question, analyze it, and think, what information do you need to answer this question, and generate related sub-questions for rag search. Do not generate sub-questions which uses the answer given by other sub-questions. \n\n Qustion: {main_question}"
        
        raw_response = self.call_api_planner(user_message=generate_subquestion_user_prompt, system_message=generate_subquestion_system_prompt)
        sub_questions_json = json.loads(raw_response.choices[0].message.content)
        
        return raw_response, sub_questions_json
    
    def generate_new_sub_questions(self, main_question, reflection_list):
        
        generate_subquestion_system_prompt = f"You are an expert reasoning assistant that breaks down a user question into concise, open-ended sub-questions. Rules:\n\nOutput only valid JSON.\n\nJSON must start with \"reasoning\" (a short paragraph on why these sub-questions are generated and their order).\n\nThen list \"1\", \"2\", \"3\", … as sub-questions..\n\nKeep sub-questions short (≤20 words).\n\nNo extra text outside the JSON."
        
        generate_subquestion_user_prompt = f"Given a question, analyze it, and think, what information do you need to answer this question, and generate related sub-questions for rag search. Do not generate sub-questions which uses the answer given by other sub-questions. \n\n Qustion: {main_question}"
        
        raw_response = self.call_api_planner(user_message=generate_subquestion_user_prompt, system_message=generate_subquestion_system_prompt)
        sub_questions_json = json.loads(raw_response.choices[0].message.content)
        
        return raw_response, sub_questions_json
    
    def generate_answer(self, sub_question, extra_information=None):

        generate_answers_system_prompt = "You are an expert reasoning assistant that generate answer of a question. Output only valid JSON."
        input_message = f"Think step by step, and answer the following question, and write both of them in Json format. Write all of your reasonings with the key 'thinking', and write the answer with the key 'answer'. Do not add detailed illustration to the answer, but add all the reasonings and explanations to the 'thinking' part of your output. If you think you can't answer the question, or the question is unanswerable, write 'I don't Know' or 'The answer is unanswerable' straight forward.\nQuestion: {sub_question}"
        
        if extra_information != None:
            input_message += f'\n\nHere are some extra information to help you answering the question.\n\nExtra information:\n{extra_information}'
            
        raw_response = self.call_api_planner(user_message=input_message, system_message=generate_answers_system_prompt)
        answer = json.loads(raw_response.choices[0].message.content)
        new_answer = "{"
        new_answer += f'"question": "{sub_question}", "thinking": "{answer["thinking"]}", "answer": "{answer["answer"]}"'
        new_answer += "}"
        new_answer = json.loads(new_answer)
        return raw_response, new_answer
    
    def generate_reflection(self, qa_pair):
        generate_reflection_prompt_system_prompt = "You are an expert reasoning assistant. Your task is to reflect on the quality of a given pair of question and answer. Think carefully. Output only valid JSON."
        input_message = f"""You are an expert reasoning assistant. Your task is to reflect on the quality of the given pair of question and answer. 
        
        Generate the reflection from the following aspects:
        - Is the given answer answered the question properly?
        - Is the reasoning closely related to the question?
        
        Only return a valid JSON object with a 'reflection' field.
        
        After reflection, based on the reflection you have generated, telll me that is the answer field answering the question correctly? Return either True or False in the JSON object with a 'TF_reflection' field. Only return True when the reflection think that the answer is ground true. In any other situations, for example, the answer is false, the question is unanserable, or the answerer don't know the answer, return false. 
        ---input---
        Question:
        {qa_pair['question']}
        
        Original reasoning:
        {qa_pair['thinking']}
        
        Predicted answer:
        {qa_pair['answer']}
        """
        raw_response = self.call_api_reflector(system_message=generate_reflection_prompt_system_prompt, user_message=input_message)
        reflection_response = json.loads(raw_response.choices[0].message.content)
        print(reflection_response)
        qa_reflection = {}
        qa_reflection['question'] = qa_pair["question"]
        qa_reflection['thinking'] = qa_pair["thinking"]
        qa_reflection['answer'] = qa_pair["answer"]
        qa_reflection['reflection'] = reflection_response["reflection"]
        qa_reflection['TF_reflection'] = reflection_response["TF_reflection"]
        # qa_reflection += f'"question": "{qa_pair["question"]}", "thinking": "{qa_pair["thinking"]}", "answer": "{qa_pair["answer"]}", "reflection": "{reflection_response["reflection"]}", "TF_reflection": {reflection_response["TF_reflection"]}'
        
        qa_reflection_json = json.loads(json.dumps(qa_reflection))
        
        return raw_response, qa_reflection_json
    
    
    def rewrite_query(self, qa_pair_for_fix, support_info):
        rewrite_query_system_message = "You are an expert reasoning assistant. Your task is to reconstruct the given question with the support information. Think carefully. Output only valid JSON."
        
        rewrite_query_input_message = f"""With the given support information, think step by step, and rewrite the original question. Generate your reasonings and the rewrited question in valid JSON. Generate your reasonings in the JSON with the key "thinking" and generate the question you rewrited with the key "new_question" 
        
        Original question:
        {qa_pair_for_fix['question']}
        
        Original question pair:
        {qa_pair_for_fix}
        
        support information:
        {support_info}
        """
        
        raw_response = self.call_api_planner(system_message=rewrite_query_system_message, user_message=rewrite_query_input_message)
        response = raw_response.choices[0].message.content
        
        response_json = json.loads(response)
        
        return raw_response, response_json
    
    def final_answer_analyze(self, question, supporting_information):
        system_prompt ="You are an expert reasoning assistant. Your task is to generate the final answer of a given query. Think carefully. Output only valid JSON."
        user_input = f"""With a given question and supporting information, think step by step, and generate both your reasoning and the final answer in valid JSON. Generate your reasonings in the key "thinking" and your final answer in the key "final_answer". Do not add detailed illustration to the answer, add all your reasonings to the "thinking" part of your output. 
        
        Question:
        {question}
        
        Supporting information:
        {supporting_information}
        """
        
        raw_response = self.call_api_planner(system_message=system_prompt, user_message=user_input)
        
        response = raw_response.choices[0].message.content
        
        response_json = json.loads(response)
        
        final_answer = response_json['final_answer']
        
        return raw_response, response_json, final_answer
    
    def execute(self, input_question):
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
        
        for x in subquestion_list:
            qa_raw_answer, qa_answer = self.generate_answer(sub_question=x)
            answered_qa_list.append(qa_answer)
        print(answered_qa_list)
        
        reflected_true_list = []
        reflected_false_list = []
        
        for x in answered_qa_list:
            raw_qa_pair_reflection, qa_pair_reflection = self.generate_reflection(qa_pair=x)
            print(qa_pair_reflection['TF_reflection'])
            if bool(qa_pair_reflection['TF_reflection']) == True:
                reflected_true_list.append(qa_pair_reflection)
            elif bool(qa_pair_reflection['TF_reflection']) == False:
                reflected_false_list.append(qa_pair_reflection)
        print(f"reflected true: \n{reflected_true_list}\n\n")
        print(f"reflected false: \n{reflected_false_list}")
        
        loop_time = 3
        
        for i in range(loop_time):
            print(f"This is loop {i}")
            new_answered_qa_list = []
            new_questions = []
            
            if reflected_false_list:
                for x in reflected_false_list:
                    raw_new_question, new_question = self.rewrite_query(qa_pair_for_fix=x, support_info=reflected_true_list)
                    new_questions.append(new_question['new_question'])
                print(new_questions)
            
                reflected_false_list = []
                
                for x in new_questions:
                    qa_raw_answer, qa_answer = self.generate_answer(sub_question=x, extra_information=reflected_true_list)
                    new_answered_qa_list.append(qa_answer)
                
                print(new_answered_qa_list)
                
                for x in new_answered_qa_list:
                    raw_qa_pair_reflection, qa_pair_reflection = self.generate_reflection(qa_pair=x)
                    print(qa_pair_reflection['TF_reflection'])
                    if bool(qa_pair_reflection['TF_reflection']) == True:
                        reflected_true_list.append(qa_pair_reflection)
                    elif bool(qa_pair_reflection['TF_reflection']) == False:
                        reflected_false_list.append(qa_pair_reflection)
        
        print(f"reflected true: \n{reflected_true_list}\n\n")
        print(f"reflected false: \n{reflected_false_list}")
        
        raw_response, response_json, final_answer = self.final_answer_analyze(question=input_question, supporting_information=reflected_true_list)
        
        print(response_json)
        print(final_answer)