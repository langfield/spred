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
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])
    print("Permutation indices:", index.eval())

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = tf.logical_not(tf.logical_or(
    tf.equal(inputs, SEP_ID),
    tf.equal(inputs, CLS_ID)))
    print("Non functional tokens:", non_func_tokens.eval())
    
    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    print("Non masked tokens:", non_mask_tokens.eval())
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)
    print("masked or func tokens:", masked_or_func_tokens.eval())
    
    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won"t be information leak
    smallest_index = -tf.ones([seq_len], dtype=tf.int64)
    print("smallest_index:", smallest_index.eval())
    rev_index = tf.where(non_mask_tokens, smallest_index, index)
    print("rev_index:", rev_index.eval())
    
    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
    print("target_tokens:", target_tokens.eval())
    target_mask = tf.cast(target_tokens, tf.float32)
    print("target_mask:", target_mask.eval())
    
    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
    print("self_rev_index:", self_rev_index.eval())
    
    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
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
        perm_size = 6
        seq_len = 6
        _local_perm(inputs, targets, is_masked, perm_size, seq_len)
        
