import torch
import numpy as np
import pandas as pd
from modeling_openai import OpenAIGPTModel

plot = False

# Create a sample tensor (this is just part of sin.csv)
tokens_tensor = torch.tensor([
    [10.0,0.0],
    [10.099833416646828,0.0],
    [10.198669330795061,0.0],
    [10.29552020666134,0.0],
    [10.38941834230865,0.0],
    [10.479425538604204,0.0],
    [10.564642473395036,0.0],
    [10.644217687237692,0.0],
    [10.717356090899523,0.0],
    [10.783326909627483,0.0],
    [10.841470984807897,0.0],
    [10.891207360061436,0.0],
    [10.932039085967226,0.0],
    [10.963558185417194,0.0]
])

width = 100
num_steps = 10
# x vals
time = np.arange(0, width, 100 / num_steps)
print('Number of data points:', time.shape[0])
# y vals
price = np.sin(time) + 10

if plot:
    plt.plot(time, price)
    plt.title('Sample Time Series')
    plt.xlabel('Time (min)')
    plt.ylabel('Price')
    plt.show()

zeros = np.ones(num_steps)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(60)]]
tokens_tensor = torch.Tensor(np.array(df))
tokens_tensor = tokens_tensor[:1000]
print("token tensor shape:", tokens_tensor.shape)
position_ids = torch.arange(0, tokens_tensor.shape[0])

id_tensor = torch.reshape(torch.arange(tokens_tensor.shape[0]), (1, tokens_tensor.shape[0]))
tokens_tensor = torch.reshape(tokens_tensor, (1, tokens_tensor.shape[0], tokens_tensor.shape[1]))

# load in our pretrained model
model = OpenAIGPTModel.from_pretrained('checkpoints_sin/')

# Set the model to evaluation mode
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(id_tensor, position_ids, None, None, tokens_tensor)
    predictions = outputs[0]

# get the predicted next token
print('Prediction: ', predictions[0, -1, :])
