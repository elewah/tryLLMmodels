# https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
# Setting `pad_token_id` to `eos_token_id`:11 for open-end generation.
# The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, 
# this will be deprecated in the future in favor of the `BetterTransformer` API. 
# Please install the latest optimum library with `pip install -U optimum` and 
# call `model.to_bettertransformer()` to benefit 
# from `torch.scaled_dot_product_attention` 
# and future performance optimizations.

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # trust_remote_code=True,
    device_map="auto",
)
iteration_times = []
# outputs = model.generate(**inputs, use_cache=True, max_length=4096)
for i in range(5):
    start_time = time.time()
    sequences = pipeline(
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
    end_time = time.time()
    iteration_time = end_time - start_time
    iteration_times.append(iteration_time)
    print(f"Iteration {i+1}: Time taken - {iteration_time} seconds")

# Calculate and print the average time taken for each iteration
average_time = sum(iteration_times) / len(iteration_times)
print(f"\nAverage time per iteration: {average_time} seconds")
