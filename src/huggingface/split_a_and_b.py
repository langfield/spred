import numpy as np
import random

"""
    REFER TO `create_tfrecords.py`. 
    `data`: a np.array of token ids with shape `(data_len,)` (one row in batch).  
    `sent_ids`: a np.array of booleans for one input_path. It alternates between `True` and `False`
        on line/sentence breaks, and when a document ends if we are using EOD. 
        Has shape `(data_len,)`.
    `begin_idx`: an integer representing where we are along `data`. Initialized
        to `reuse_len`. Incremented by `reuse_len` on each call to this function.
    `tot_len`: integer set to `seq_len` - `reuse_len` - 3. Constant across all calls.   
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
    # Checking if `end_idx` is the first pos in a new sentence.
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
      cut_points.append(end_idx) # `cut_points` is all the first positions of sents. 
    end_idx += 1

  # `end_idx` is now equal to `begin_idx` + `tot_len`.
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
    print("begin_idx:", begin_idx) 
    print("end_idx:", end_idx) 
    print("tot_len:", tot_len)
    print("a_end - a_begin:", a_end - a_begin) 
    print("b_len:", b_len) 
    print("b_begin:", b_begin) 
    print("b_end:", b_end) 
    #===MOD===
    return
    #===MOD===
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

if __name__ == "__main__":
    data = [i for i in range(60)]
    sent_ids = [True for i in range(60)]
    data = np.array(data)
    sent_ids = np.array(sent_ids)
    begin_idx = 3
    tot_len = 3
    _split_a_and_b(data, sent_ids, begin_idx, tot_len, True)
