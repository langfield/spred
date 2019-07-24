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
    
    # Generate permutation indices
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])
    
    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = tf.logical_not(tf.logical_or(
    tf.equal(inputs, SEP_ID),
    tf.equal(inputs, CLS_ID)))
    
    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)
    
    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
   
    # EXAMPLE
    # `seq_len` is 5.  
    # `index` is a permutation of the form [4, 2, 1, 0, 3].
    # `smallest_index` is a tensor of the form [-1, -1, -1, -1, -1]. 
    # `non_mask_tokens` is a tensor of booleans of the form [True, True, True, True, False]
    # `rev_index` is a copy of index where all positions which are `False` in `non_mask_tokens`
    #       are set to -1. 
    # `rev_index` is thus a tensor of the form [4, 2, 1, 0, -1]. 
    smallest_index = -tf.ones([seq_len], dtype=tf.int64) # Just all -1s
    rev_index = tf.where(non_mask_tokens, smallest_index, index)
    # In `rev_index` the non-masked tokens are -1.    
 
    # Create `target_mask`: non-functional and masked tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens) # Gets masked, non-functional tokens. 
    target_mask = tf.cast(target_tokens, tf.float32)
    
    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    
    # The -1s are the non-masked, non-functional, the target tokens are masked, non-functional. 
    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
    # `self_rev_index` has original indices for masked, non-functional tokens, 0 for non-masked tokens, and original indices + 1 for masked, functional tokens. 
    
    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
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
