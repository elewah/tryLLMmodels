# https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "Upstage/SOLAR-10.7B-Instruct-v1.0",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

conversation = [ {'role': 'user', 'content': 'Hello?'} ] 

prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
model.generation_config.max_length = 10
# outputs = model.generate(**inputs, use_cache=True, max_length=4096)
for i in range(5):
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    output_text = tokenizer.decode(outputs[0]) 
    print(output_text)


