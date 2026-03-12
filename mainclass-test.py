from openai import OpenAI
import huggingface_hub
import os
import json
from mainclass_hf import multi_reflection_rag_hf
from mainclass_openaisdk import multi_reflection_rag
from transformers import AutoModelForCausalLM, AutoTokenizer

huggingface_hub.login(token=hf_token)

# Qwen3-8b huggingface
qwen3_modelname = "Qwen/Qwen3-8B"
qwen3_8b_tokenizer = AutoTokenizer.from_pretrained(qwen3_modelname)
qwen3_model = AutoModelForCausalLM.from_pretrained(
    qwen3_modelname,
    torch_dtype="auto",
    device_map="auto"
)

# openai client
openai_gpt_client = OpenAI(api_key=openai_api)
# models: gpt-5

# deepseek client
deepseek_client = OpenAI(api_key=ds_api, base_url="https://api.deepseek.com") 
# models: deepseek-chat, deepseek-reasoner

# gemini client
gemini_client = OpenAI(api_key=gemini_api, base_url="https://generativelanguage.googleapis.com/v1beta/openai/") 
# models: gemini-2.5-flash

# qwen client
qwen_client = OpenAI(api_key=qwen_api, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
# model: qwen-plus

# kimi client
kimi_client = OpenAI(api_key=kimi_api, base_url="https://api.moonshot.cn/v1")
# models: kimi-k2-0905-preview
    
runclass = multi_reflection_rag(
    planner_client=qwen_client,
    reflector_client=gemini_client,
    planner_model="qwen-plus",
    reflector_model="gemini-2.5-flash",
)

runclass.execute(input_question="What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?")