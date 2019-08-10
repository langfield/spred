import os
import torch
import numpy as np
import pandas as pd
from modeling_openai import OpenAIGPTModel
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
#===MOD===
from torch.autograd import Variable
#===MOD===

plot = False

width = 100
num_steps = 10000
# x vals
time = np.arange(0, width, 100 / num_steps)
print('Number of data points:', time.shape[0])
# y vals
price = np.sin(time) + 10
# price = np.array([0.5] * num_steps)


if plot:
    plt.plot(time, price)
    plt.title('Sample Time Series')
    plt.xlabel('Time (min)')
    plt.ylabel('Price')
    plt.show()

zeros = np.ones(num_steps)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(60)]]
# print(df)
tokens_tensor = torch.Tensor(np.array(df))
tokens_tensor = tokens_tensor[:30]
inputs_raw = tokens_tensor.cuda()

position_ids = torch.arange(0, tokens_tensor.shape[0])
position_ids = torch.stack([position_ids])
id_tensor = position_ids
print("token tensor shape:", tokens_tensor.shape)
print("position_ids shape:", position_ids.shape)

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
