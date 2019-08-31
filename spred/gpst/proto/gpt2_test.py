import torch
from modeling_gpt2 import GPT2Config, GPT2Model

config = GPT2Config.from_pretrained('gpt2')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model(config)
input_ids = torch.tensor([[15496,    11,   616,  3290,   318, 13779]])
print('input_ids:', input_ids)
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(outputs[0])