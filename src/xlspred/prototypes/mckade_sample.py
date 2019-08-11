import copy
import json
import torch
from pytorch_transformers import *

print('Getting config...')
config = XLNetConfig.from_pretrained('config.json')
print("Config file:", config)
# config.save_pretrained("./")

print('Getting tokenizer...')
#tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
print('Getting model...')
model = XLNetModel(config)
print('Tokenizing input...')
print('Tokenizer encode output:', tokenizer.encode("Hello, my dog is cute"))
seq = tokenizer.encode("Hello, my dog is cute")
input_ids = torch.tensor([seq, copy.deepcopy(seq)])  # Batch size 1
input_ids = torch.tensor(
    [[[ 0.0036,  0.0227],
     [ 0.0036,  0.0227]],
    [[ 0.0128,  0.0058],
     [ 0.0128,  0.0058]],
    [[ 0.0315,  0.0126],
     [ 0.0315,  0.0126]],
    [[ 0.0096, -0.0011],
     [ 0.0096, -0.0011]],
    [[ 0.0066,  0.0123],
     [ 0.0066,  0.0123]],
    [[ 0.0006,  0.0043],
     [ 0.0006,  0.0043]],
    [[-0.0346,  0.0088],
     [-0.0346,  0.0088]]]
    )
# size 1
print("input_ids:", input_ids)
print('Getting output...')
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)
