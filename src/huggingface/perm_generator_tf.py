import tensorflow as tf
import os

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
    print("Permutation indices:", index.eval())

    # `perm_mask` and `target_mask`
    # non-functional tokens
    """
    Very straightforward: everything that isn't functional.     
    """
    non_func_tokens = tf.logical_not(tf.logical_or(
    tf.equal(inputs, SEP_ID),
    tf.equal(inputs, CLS_ID)))
    print("Non functional tokens:", non_func_tokens.eval())
   
    """
    Non-masked AND non-functional tokens. 
    """ 
    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    print("Non masked tokens:", non_mask_tokens.eval())
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)
    print("masked or func tokens:", masked_or_func_tokens.eval())
    
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
    print("smallest_index:", smallest_index.eval())

    """
    Gets the indices of the sequence via `index`, and sets all the
    non-masked and non-functional tokens to `-1`.     
    """
    rev_index = tf.where(non_mask_tokens, smallest_index, index)
    print("rev_index:", rev_index.eval())
    
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
    print("target_tokens:", target_tokens.eval())
    target_mask = tf.cast(target_tokens, tf.float32)
    print("target_mask:", target_mask.eval())
    
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
    print("self_rev_index:", self_rev_index.eval())
    
    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    """
    `1` marks tokens we're masking, `0` marks tokens we are not masking. 
    The token we are currently at (trying to predict) is at `i`, and the one we're trying to attend to is
    at `j`. If `i <= j` in the permutation order, then we cannot see it, and if j is masked, we cannot see it. 
    Otherwise, we can see it.  
    """
    print("self_rev_index[:, None]:", self_rev_index[:, None].eval())
    print("rev_index[None, :]:", rev_index[None, :].eval())
    print("<=", (self_rev_index[:, None] <= rev_index[None, :]).eval())
    perm_mask = tf.logical_and(
    self_rev_index[:, None] <= rev_index[None, :],
    masked_or_func_tokens)
    print("perm_mask:", perm_mask.eval())
    perm_mask = tf.cast(perm_mask, tf.float32)
    print("perm_mask:", perm_mask.eval())
    
    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = tf.concat([inputs[0: 1], targets[: -1]],
    axis=0)
    print("new_targets:", new_targets.eval())
    
    # construct inputs_k
    inputs_k = inputs
    
    # construct inputs_q
    inputs_q = target_mask
    
    return perm_mask, new_targets, target_mask, inputs_k, inputs_q

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    sess = tf.Session()
    with sess.as_default():
        inputs = tf.constant([3,4,5,6,7,8])
        targets = tf.constant([3,4,5,6,7,8])
        is_masked = tf.constant([False, False, False, False, False, True], bool)
        perm_size = 1
        seq_len = 6
        _local_perm(inputs, targets, is_masked, perm_size, seq_len)
        
