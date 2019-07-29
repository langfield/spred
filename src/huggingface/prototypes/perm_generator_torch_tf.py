import os
import torch
import numpy as np
import tensorflow as tf
np.random.seed(89)

DEBUG = False

def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
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
    index = torch.cat(perm_portions, 0).cuda()
    
    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = ~(inputs.eq(SEP_ID) | inputs.eq(CLS_ID))
   
    """
    Non-masked AND non-functional tokens. 
    """ 
    non_mask_tokens = (~is_masked & non_func_tokens).cuda()
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
    smallest_index = torch.IntTensor([-1 for i in range(seq_len)]).cuda()

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
        print("new_targets:", new_targets)
    
    # construct inputs_k
    inputs_k = inputs
    
    # construct inputs_q
    inputs_q = target_mask
    
    return perm_mask, new_targets, target_mask, inputs_k, inputs_q

def _local_perm_tf(inputs, targets, is_masked, perm_size, seq_len):
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

        reuse_len: the number of tokens in the currect batch to be cached
      and reused in the future.
    """

    SEP_ID = 3
    CLS_ID = 4

    # Generate permutation indices
    """
    Computes a random permutation. We must have that
        seq_len % perm_size == 0. 
    So in particular, we must have that 
        1 <= perm_size <= seq_len.
    If `perm_size` is strictly smaller than `seq_len`, then
    it just repeats the permutation `seq_len / perm_size` times. 
    
    EXAMPLE:
        seq_len = 6
        perm_size = 3
        permutation = (1 3 2)
        result = [0 2 1 3 5 4]
    
    So the way to implement this is to random shuffle `perm_size`
    elements, and then apply the resulting permutation to each
    section of the sequence (0, 1, 2,...,k).    
    """
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    # Numpy version. 
    np.random.seed(59)
    perm = np.random.permutation(perm_size)
    repeats = int(seq_len / perm_size)
    perm_portions = [tf.constant(perm + i * perm_size) for i in range(repeats)] 
    index = tf.concat(perm_portions, axis=0)

    # print("Permutation indices:", copy_index.eval())

    # `perm_mask` and `target_mask`
    # non-functional tokens
    """
    Very straightforward: everything that isn't functional.     
    """
    non_func_tokens = tf.logical_not(tf.logical_or(
    tf.equal(inputs, SEP_ID),
    tf.equal(inputs, CLS_ID)))
   
    """
    Non-masked AND non-functional tokens. 
    """ 
    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)
    
    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    """
    An array of length `seq_len` of all `-1`s. 
    
    EXAMPLE:
        [-1, -1, -1, -1, -1].
    """
    smallest_index = -tf.ones([seq_len], dtype=tf.int64)

    """
    Gets the indices of the sequence via `index`, and sets all the
    non-masked and non-functional tokens to `-1`.     
    """
    rev_index = tf.where(non_mask_tokens, smallest_index, index)
    
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
    target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
    target_mask = tf.cast(target_tokens, tf.float32)
    
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
    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
    
    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    """
    `1` marks tokens we're masking, `0` marks tokens we are not masking. 
    The token we are currently at (trying to predict) is at `i`, and the one we're trying to attend to is
    at `j`. If `i <= j` in the permutation order, then we cannot see it, and if j is masked, we cannot see it. 
    Otherwise, we can see it.  
    """
    perm_mask = tf.logical_and(
    self_rev_index[:, None] <= rev_index[None, :],
    masked_or_func_tokens)
    perm_mask = tf.cast(perm_mask, tf.float32)
    
    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = tf.concat([inputs[0: 1], targets[: -1]],
    axis=0)
    
    # construct inputs_k
    inputs_k = inputs
    
    # construct inputs_q
    inputs_q = target_mask
    
    return perm_mask, new_targets, target_mask, inputs_k, inputs_q

if __name__ == "__main__":
    perm_size = 40
    seq_len = 800
    input_array = np.random.permutation(seq_len) # Numpy version.
    target_array = np.random.permutation(seq_len) # Numpy version.
    bool_array = np.random.permutation(seq_len) # Numpy version.
    
    inputs = torch.IntTensor(input_array)
    targets = torch.IntTensor(target_array)
    is_masked = torch.Tensor([0 if i % 2 == 0 else 1 for i in bool_array]).byte()
   
    # print("inputs:", inputs)
    # print("targets:", targets)
    # print("is_masked:", is_masked)

    perm_mask, new_targets, target_mask, inputs_k, inputs_q = _local_perm(inputs, targets, is_masked, perm_size, seq_len)

    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    sess = tf.Session()
    with sess.as_default():
        inputs = tf.constant(input_array)
        targets = tf.constant(target_array)
        is_masked = tf.constant([False if i % 2 == 0 else True for i in bool_array], bool)
        # print("inputs_tf:", inputs.eval())
        # print("targets_tf:", targets.eval())
        # print("is_masked_tf:", is_masked.eval())
    
        perm_mask_tf, new_targets_tf, target_mask_tf, inputs_k_tf, inputs_q_tf = _local_perm_tf(inputs, targets, is_masked, perm_size, seq_len)

        print(perm_mask.shape)
        print(perm_mask_tf.shape)
        # print("final perm_mask_torch:\n", np.array(perm_mask))
        # print("final perm_mask_tf:\n", np.array(perm_mask_tf.eval()))
        np.testing.assert_almost_equal(np.array(perm_mask), np.array(perm_mask_tf.eval()))
        np.testing.assert_almost_equal(np.array(new_targets), np.array(new_targets_tf.eval()))
        np.testing.assert_almost_equal(np.array(target_mask), np.array(target_mask_tf.eval()))
        np.testing.assert_almost_equal(np.array(inputs_k), np.array(inputs_k_tf.eval()))
        np.testing.assert_almost_equal(np.array(inputs_q), np.array(inputs_q_tf.eval()))
