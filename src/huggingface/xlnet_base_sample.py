import json
import torch
from pytorch_transformers import *

print('Getting config...')
config = XLNetConfig.from_pretrained('xlnet-base-cased')
print("Config file:", config)
# config.save_pretrained("./")

print('Getting tokenizer...')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
print('Getting model...')
model = XLNetModel(config)
print('Tokenizing input...')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
print("input_ids:", input_ids)
print('Getting output...')
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)
