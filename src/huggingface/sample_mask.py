def _sample_mask(sp, seg, reverse=False, max_gram=5, goal_num_predict=None):
  """Sample `goal_num_predict` tokens for partial prediction.
  About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

  seg_len = len(seg)
  mask = np.array([False] * seg_len, dtype=np.bool)

  num_predict = 0

  ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, max_gram + 1)
  pvals /= pvals.sum(keepdims=True)

  if reverse:
    seg = np.flip(seg, 0)

  cur_len = 0
  while cur_len < seg_len:
    if goal_num_predict is not None and num_predict >= goal_num_predict: break

    n = np.random.choice(ngrams, p=pvals)
    if goal_num_predict is not None:
      n = min(n, goal_num_predict - num_predict)
    ctx_size = (n * FLAGS.mask_alpha) // FLAGS.mask_beta
    l_ctx = np.random.choice(ctx_size)
    r_ctx = ctx_size - l_ctx

    # Find the start position of a complete token
    beg = cur_len + l_ctx
    while beg < seg_len and not _is_start_piece(sp.IdToPiece(seg[beg].item())):
      beg += 1
    if beg >= seg_len:
      break

    # Find the end position of the n-gram (start pos of the n+1-th gram)
    end = beg + 1
    cnt_ngram = 1
    while end < seg_len:
      if _is_start_piece(sp.IdToPiece(seg[beg].item())):
        cnt_ngram += 1
        if cnt_ngram > n:
          break
      end += 1
    if end >= seg_len:
      break

    # Update
    mask[beg:end] = True
    num_predict += end - beg

    cur_len = end + r_ctx

  while goal_num_predict is not None and num_predict < goal_num_predict:
    i = np.random.randint(seg_len)
    if not mask[i]:
      mask[i] = True
      num_predict += 1

  if reverse:
    mask = np.flip(mask, 0)

  return mask
