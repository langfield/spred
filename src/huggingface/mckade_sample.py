import copy
import json
import torch
from pytorch_transformers import *

print('Getting config...')
config = XLNetConfig.from_pretrained('config.json')
print("Config file:", config)
# config.save_pretrained("./")

print('Getting tokenizer...')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
print('Getting model...')
model = XLNetModel(config)
print('Tokenizing input...')

print('Tokenizer encode output:', tokenizer.encode("Hello, my dog is cute"))
seq = tokenizer.encode("Hello, my dog is cute")
input_ids = torch.tensor([seq, copy.deepcopy(seq)])  # Batch size 1
input_ids =  torch.tensor([[[-0.0050, -0.0232],
         [-0.0050, -0.0232]],

        [[ 0.0358, -0.0225],
         [ 0.0358, -0.0225]],

        [[ 0.0110,  0.0015],
         [ 0.0110,  0.0015]],

        [[ 0.0368,  0.0127],
         [ 0.0368,  0.0127]],

        [[ 0.0081, -0.0139],
         [ 0.0081, -0.0139]],

        [[-0.0427, -0.0104],
         [-0.0427, -0.0104]],

        [[ 0.0076, -0.0234],
         [ 0.0076, -0.0234]]])
print("input_ids:", input_ids)
print('Getting output...')
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states)
