import os
import torch
import numpy as np
DEBUG = True

def _local_perm(inputs, targets, is_masked, perm_size, seq_len, device):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.
    Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
        for partial prediction.
        perm_size: the length of longest permutation. Could be set to be reuse_len.
        Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
    """
    
    SEP_ID = 3
    CLS_ID = 4
    
    # Generate permutation indices
    assert seq_len % perm_size == 0
    perm = torch.randperm(perm_size).int()
    np.random.seed(59)
    perm = torch.IntTensor(np.random.permutation(perm_size)) # Numpy version.
    repeats = int(seq_len / perm_size)
    perm_portions = [perm + i * perm_size for i in range(repeats)]
    index = torch.cat(perm_portions, 0).to(device)
    
    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = ~(inputs.eq(SEP_ID) | inputs.eq(CLS_ID))
   
    """
    Non-masked AND non-functional tokens. 
    """ 
    non_mask_tokens = (~is_masked & non_func_tokens).to(device)
    masked_or_func_tokens = ~non_mask_tokens
      
    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    """
    An array of length `seq_len` of all `-1`s. 
    
    EXAMPLE:
        [-1, -1, -1, -1, -1].
    """
    smallest_index = torch.IntTensor([-1 for i in range(seq_len)]).to(device)

    """
    Gets the indices of the sequence via `index`, and sets all the
    non-masked and non-functional tokens to `-1`.     
    """
    rev_index = torch.where(non_mask_tokens, smallest_index, index)
    
    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    """
    Masked or functional tokens and non-functional tokens. 
    So these must be non-functional. 
    Tokens in `masked_or_func_tokens` must be either masked, functional, or both. 
    Thus if these are non-functional tokens, they must be masked. 
    So they are exactly all the masked and non-functional tokens. 
    """
    target_tokens = masked_or_func_tokens & non_func_tokens
    target_mask = target_tokens.float()
    
    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    """
    If a token is a target, it keeps its original index, since it is masked and non-functional, 
    so it cannot be non-masked, so it cannot be set to `-1` in `rev_index`. 
    Otherwise, we increment the value from `rev_index`.
    These are all non-masked tokens, and all functional tokens. If a token
    happens to be non-masked and non-functional, it will be incremented from `-1` to `0`. 
    Thus there will be no `-1`s in the final tensor. 
    """
    self_rev_index = torch.where(target_tokens, rev_index, rev_index + 1)
    
    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    """
    `1` marks tokens we're masking, `0` marks tokens we are not masking. 
    The token we are currently at (trying to predict) is at `i`, and the one we're trying to attend to is
    at `j`. If `i <= j` in the permutation order, then we cannot see it, and if j is masked, we cannot see it. 
    Otherwise, we can see it.  
    """
    perm_mask = (self_rev_index[:, None] <= rev_index[None, :]) & masked_or_func_tokens
    perm_mask = perm_mask.float()
    # print("perm_mask:\n", np.array(perm_mask))
    
    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = torch.cat([inputs[0: 1], targets[: -1]], 0)
    
    if DEBUG:
        print("permutation index:", np.array(index))
        print("Non functional tokens:", np.array(non_func_tokens))
        print("Non masked tokens:", np.array(non_mask_tokens))
        print("masked or func tokens:", np.array(masked_or_func_tokens))
        print("smallest_index:\n", np.array(smallest_index))
        print("rev_index:\n", np.array(rev_index))
        print("target_tokens:\n", np.array(target_tokens))
        print("target_mask:\n", np.array(target_mask))
        print("self_rev_index:\n", np.array(self_rev_index))
        print("self_rev_index[:, None]:\n", np.array(self_rev_index[:, None]))
        print("rev_index[None, :]:\n", np.array(rev_index[None, :]))
        print("<=\n", np.array(self_rev_index[:, None] <= rev_index[None, :]))
        print("perm_mask:\n", np.array(perm_mask))
        print("inputs[0: 1]:", inputs[0: 1])
        print("targets[: -1]:", targets[: -1])
        print("new_targets:", new_targets)
    
    # construct inputs_k
    inputs_k = inputs
    
    # construct inputs_q
    inputs_q = target_mask
    
    return perm_mask, new_targets, target_mask, inputs_k, inputs_q
