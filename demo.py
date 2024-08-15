import os
os.chdir("../")

from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch
from src import *

model_size = "1.4b"

model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{model_size}-deduped",
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped",
  revision="step3000",
  cache_dir="./pythia-160m-deduped/step3000",
  device_map="auto"
)

reverse_model = GPTNeoXForCausalLM.from_pretrained(
    "afterless/reverse-pythia-160m"
).cuda()
input_str = " !" * 15
expected_output = " should never be president"

def generate_from_reversal(output):
    return tokenizer.decode(
        model.generate(
            input_ids=tokenizer.encode(
                output.replace(expected_output, ""),
                return_tensors="pt").cuda(),
                max_new_tokens=25
        )[0],
    ).replace(output.replace(expected_output, ""), "")

rlm = ReverseModelSampler(
    model,
    reverse_model,
    tokenizer,
    num_beams=10
)
output4 = rlm.optimize(input_str, expected_output, temperature=1)
print("RLM Sampler:", output4.replace("\n", ""))

print(generate_from_reversal(output4))