from create_tfrecords import batchify
import torch

orig_data_len = 50
batch_size = 5
data = torch.arange(0, orig_data_len)
new_data = batchify(data, batch_size)
print("orig_data_len:", orig_data_len)
print("batch_size:", batch_size)
print("New_data:\n", new_data)
