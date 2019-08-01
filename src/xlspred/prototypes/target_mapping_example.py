import numpy as np
import torch

# Params. 
SEQ_LEN = 7

# Equivalent to ``tf.one_hot(indices, seq_len)``. 
def one_hot(indices: torch.ByteTensor, seq_len: int) -> torch.FloatTensor:
    index_len = len(indices)
    inp = indices % seq_len
    inp_ = torch.unsqueeze(inp, 1)
    one_hot = torch.FloatTensor(index_len, seq_len).zero_()
    one_hot.scatter_(1, inp_, 1)
    return one_hot

if __name__ == "__main__":
    
    # Setup.
    perm = np.random.permutation(SEQ_LEN)
    bools = [True if elem % 2 == 0 else False for elem in perm]
    indices = torch.arange(0, SEQ_LEN)
    bool_target_mask = torch.Tensor(bools).byte()
    indices = indices[bool_target_mask]

    # Function call. 
    one_hot = one_hot(indices, SEQ_LEN)
    
    # Output. 
    print("Seq length:", SEQ_LEN)
    print("Bool target mask:", bool_target_mask)
    print("Original indices:", torch.arange(0, SEQ_LEN))
    print("Masked indices:", indices)
    print("Output one_hot:\n", one_hot)
