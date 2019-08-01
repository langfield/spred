import numpy as np

MASK_ALPHA = 6
MASK_BETA = 1
"""
Inputs:
    **seg**: ``torch.Tensor`` of shape ``(reuse_len,)`` or ``(seq_len - reuse_len,)``:
        The segment of a single row of one batch of input_ids (character ids in XLNet). 
    **reverse**: ``bool``:
        Whether or not to reverse. 
    **max_gram**: ``int``:
        Max n-grams. 
    **goal_num_predict**: ``int``:
        Number of chars for partial prediction.  
"""
def _sample_mask(seg, reverse=False, max_gram=5, goal_num_predict=None):
  """Sample `goal_num_predict` chars for partial prediction.
  About `mask_beta` chars are chosen in a context of `mask_alpha` chars."""

  seg_len = len(seg)
  mask = np.array([False] * seg_len, dtype=np.bool) # Init. mask to all ``False``. 

  num_predict = 0

  ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, max_gram + 1)
  pvals /= pvals.sum(keepdims=True)

  if reverse:
    seg = np.flip(seg, 0)

  cur_len = 0
  while cur_len < seg_len:
    if goal_num_predict is not None and num_predict >= goal_num_predict: break

    # Grab a rand number from [1,2,...,max_gram] with equal prob. 
    n = np.random.choice(ngrams, p=pvals)
    if goal_num_predict is not None:
      n = min(n, goal_num_predict - num_predict)
    
    # We choose a window of size ``n * FLAGS.mask_alpha``, and will mask 
    # ``FLAGS.mask_beta`` chars from this window. The context is the set of all
    # chars we don't mask, thus the ``ctx_size`` is 
    #       ``(n * FLAGS.mask_alpha) // FLAGS.mask_beta``. 
    ctx_size = (n * MASK_ALPHA) // MASK_BETA # Set to 6, 1, respectively. 
    l_ctx = np.random.choice(ctx_size)  # Rand num in [0,ctx_size)
    r_ctx = ctx_size - l_ctx             
    # We are partitioning ``ctx_size``. 
    """
    EXAMPLE:
        ctx_size = 30
        l_ctx = 10
        r_ctx = 20
        
        We always have l_ctx + r_ctx = ctx_size. 
    """

    # Find the start position of a complete token (working with char_ids). 
    beg = cur_len + l_ctx
    if beg >= seg_len:
      break

    # Find the end position of the n-gram (start pos of the n+1-th gram)
    end = beg + 1
    cnt_ngram = 1
    while end < seg_len:
      # Ultimately break outer loop if seg[beg] is not beginning of an n-gram, 
      # since we must be near end of sequence. 
      cnt_ngram += 1
      if cnt_ngram > n:
        break
      end += 1
    if end >= seg_len:
      break

    # So now [beg:end] is an n-gram. 
    # Update
    mask[beg:end] = True
    num_predict += end - beg

    cur_len = end + r_ctx # Move past this context window, i.e. to next one. 

  while goal_num_predict is not None and num_predict < goal_num_predict:
    i = np.random.randint(seg_len)
    if not mask[i]:
      mask[i] = True
      num_predict += 1

  if reverse:
    mask = np.flip(mask, 0)

  return mask
