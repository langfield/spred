import numpy as np
import torch

seq_len = 7

perm = np.random.permutation(seq_len)
bools = [True if elem % 2 == 0 else False for elem in perm]
indices = torch.arange(0, seq_len)
bool_target_mask = torch.Tensor(bools).byte()
indices = indices[bool_target_mask]
index_len = len(indices)
inp = indices % seq_len
inp_ = torch.unsqueeze(inp, 1)
one_hot = torch.FloatTensor(index_len, seq_len).zero_()
one_hot.scatter_(1, inp_, 1)

print("Seq length:", seq_len)
print("Bool target mask:", bool_target_mask)
print("Original indices:", torch.arange(0, seq_len))
print("Masked indices:", indices)
print("Output one_hot:\n", one_hot)
