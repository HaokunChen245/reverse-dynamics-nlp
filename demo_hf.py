from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch as t

tokenizer = AutoTokenizer.from_pretrained(
    "afterless/reverse-pythia-160m"
)
model = GPTNeoXForCausalLM.from_pretrained(
    "afterless/reverse-pythia-160m"
)

inputs = tokenizer(
    "but I told him, the cheese was the best",
    return_token_type_ids=False,
    return_tensors="pt"
)
inputs['input_ids'] = t.flip(inputs.input_ids, (1,))
tokens = t.flip(model.generate(**inputs), (1,))
print(tokenizer.decode(tokens[0]))

model_size = "1.4b"

model = GPTNeoXForCausalLM.from_pretrained(
  f"EleutherAI/pythia-{model_size}-deduped",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-1.4b-deduped",
  revision="step3000",
  cache_dir="./pythia-160m-deduped/step3000",
  device_map="auto"
)
inputs = tokenizer(
    ". I tried to talk him out of it",
    return_token_type_ids=False,
    return_tensors="pt"
)
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))