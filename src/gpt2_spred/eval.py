import os
import torch
import numpy as np
import pandas as pd
from modeling_openai import OpenAIGPTModel
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from dataset import GPTSpredEvalDataset
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
#===MOD===
from torch.autograd import Variable
#===MOD===
try:
    from torch.utils.data import SequentialSampler
except ImportError:
    from torch_addons.sampler import SequentialSampler

# load in our pretrained model      
output_dir = 'checkpoints/'
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
loaded_config = OpenAIGPTConfig.from_json_file(output_config_file)
model = OpenAIGPTLMHeadModel(loaded_config)
model.load_state_dict(torch.load(output_model_file))

# Set the model to evaluation mode
model.cuda()
model.eval()
print("Is model training:", model.training) 

DIM = model.config.n_embd
MAX_SEQ_LEN = model.config.n_positions
BATCH_SIZE = 4
TOTAL_DATA_LEN = 10000

print("Data dimensionality:", DIM)
print("Max sequence length :", MAX_SEQ_LEN)
print("Eval batch size:", BATCH_SIZE)

plot = False

width = 100
# x vals
time = np.arange(0, width, 100 / TOTAL_DATA_LEN)
print('Number of data points:', time.shape[0])
# y vals
price = np.sin(time) + 10
# price = np.array([0.5] * TOTAL_DATA_LEN)

if plot:
    plt.plot(time, price)
    plt.title('Sample Time Series')
    plt.xlabel('Time (min)')
    plt.ylabel('Price')
    plt.show()

zeros = np.ones(TOTAL_DATA_LEN)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(DIM)]]
# print(df)
tensor_data = torch.Tensor(np.array(df))
tensor_data.view()
inputs_raw = tensor_data.cuda()

eval_data = GPTSpredEvalDataset(tensor_data, MAX_SEQ_LEN) 
print("Length of eval dataset:", len(eval_data))
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)


position_ids = torch.arange(0, tensor_data.shape[0])
position_ids = torch.stack([position_ids])
id_tensor = position_ids
print("tensor data shape:", tensor_data.shape)
print("position_ids shape:", position_ids.shape)

# Predict all tokens
input_ids = id_tensor.long().cuda()
position_ids = Variable(id_tensor.long().cuda()).contiguous()
# position_ids = Variable(id_tensor.long()).contiguous()
#===DEBUG===
print("=======================================")
print("Type of input_ids:", type(input_ids)) 
print("Type of position_ids:", type(position_ids)) 
print("type of position_ids data:", type(position_ids.data)) 
print("Type of inputs_raw:", type(inputs_raw)) 
#===DEBUG===
outputs = model(input_ids, position_ids, None, None, inputs_raw)
predictions = outputs[0]

# get the predicted next token
print('Prediction: ', predictions[0, -1, :])
