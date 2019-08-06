import torch
import numpy as np
import pandas as pd
from modeling_openai import OpenAIGPTModel

plot = False

width = 100
num_steps = 10000
# x vals
time = np.arange(0, width, 100 / num_steps)
print('Number of data points:', time.shape[0])
# y vals
# price = np.sin(time) + 10
price = np.array([0.5] * num_steps)


if plot:
    plt.plot(time, price)
    plt.title('Sample Time Series')
    plt.xlabel('Time (min)')
    plt.ylabel('Price')
    plt.show()

zeros = np.ones(num_steps)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(60)]]
print(df)
tokens_tensor = torch.Tensor(np.array(df))
tokens_tensor = tokens_tensor[:30]
position_ids = torch.arange(0, tokens_tensor.shape[0])
position_ids = torch.stack([position_ids])
id_tensor = position_ids
print("token tensor shape:", tokens_tensor.shape)
print("position_ids shape:", position_ids.shape)


# load in our pretrained model
model = OpenAIGPTModel.from_pretrained('checkpoints/')

# Set the model to evaluation mode
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(id_tensor, id_tensor, None, None, tokens_tensor)
    predictions = outputs[0]

# get the predicted next token
print('Prediction: ', predictions[0, -1, :])
