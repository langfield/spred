"""
    `data`: a single row from the dataset. 
    `sent_ids`: 
"""
def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
  """Split two segments from `data` starting from the index `begin_idx`."""

  data_len = data.shape[0]
  if begin_idx + tot_len >= data_len:
    tf.logging.info("[_split_a_and_b] returns None: "
                    "begin_idx %d + tot_len %d >= data_len %d",
                    begin_idx, tot_len, data_len)
    return None

  end_idx = begin_idx + 1
  cut_points = []
  while end_idx < data_len:
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
      cut_points.append(end_idx)
    end_idx += 1

  a_begin = begin_idx
  if len(cut_points) == 0 or random.random() < 0.5:
    label = 0
    if len(cut_points) == 0:
      a_end = end_idx
    else:
      a_end = random.choice(cut_points)

    b_len = max(1, tot_len - (a_end - a_begin))
    # (zihang): `data_len - 1` to account for extend_target
    b_begin = random.randint(0, data_len - 1 - b_len)
    b_end = b_begin + b_len
    while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
      b_begin -= 1
    # (zihang): `data_len - 1` to account for extend_target
    while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
      b_end += 1

    new_begin = a_end
  else:
    label = 1
    a_end = random.choice(cut_points)
    b_begin = a_end
    b_end = end_idx

    new_begin = b_end

  while a_end - a_begin + b_end - b_begin > tot_len:
    if a_end - a_begin > b_end - b_begin:
      # delete the right side only for the LM objective
      a_end -= 1
    else:
      b_end -= 1

  ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

  if extend_target:
    if a_end >= data_len or b_end >= data_len:
      tf.logging.info("[_split_a_and_b] returns None: "
                      "a_end %d or b_end %d >= data_len %d",
                      a_end, b_end, data_len)
      return None
    a_target = data[a_begin + 1: a_end + 1]
    b_target = data[b_begin: b_end + 1]
    ret.extend([a_target, b_target])

  return ret
