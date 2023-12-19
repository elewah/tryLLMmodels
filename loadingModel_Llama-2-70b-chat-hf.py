# https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from huggingface_hub import login
import os
from dotenv import load_dotenv
load_dotenv()
Huggingface_token= os.getenv("HuggingfaceToken")
login(Huggingface_token)

# login("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")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat")
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-chat",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# from transformers import AutoModelForCausalLM, AutoTokenizer

model_id="meta-llama/Llama-2-70b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model =AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

conversation = [ {'role': 'user', 'content': 'please fix the grammer in this line "my name is machiel he have book"'} ] 

prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
model.generation_config.max_length = 10
# Initialize an empty list to store the time taken for each iteration
iteration_times = []
# outputs = model.generate(**inputs, use_cache=True, max_length=4096)
for i in range(5):
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    output_text = tokenizer.decode(outputs[0]) 
    end_time = time.time()
    iteration_time = end_time - start_time
    iteration_times.append(iteration_time)
    print(output_text)
    print(f"Iteration {i+1}: Time taken - {iteration_time} seconds")

# Calculate and print the average time taken for each iteration
average_time = sum(iteration_times) / len(iteration_times)
print(f"\nAverage time per iteration: {average_time} seconds")


