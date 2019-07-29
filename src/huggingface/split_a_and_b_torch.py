import numpy as np
import random
import torch

DEBUG = False

"""
    REFER TO `create_tfrecords.py`. 
    `data`: a np.array of token ids with shape `(data_len,)` (one row in batch).  
    `sent_ids`: REMOVED (we will only ever have one input row) 
        a np.array of booleans for one input_path. It alternates between `True` and `False`
        on line/sentence breaks, and when a document ends if we are using EOD. 
        Has shape `(data_len,)`.
    `begin_idx`: an integer representing where we are along `data`. Initialized
        to `reuse_len`. Incremented by `reuse_len` on each call to this function.
    `tot_len`: integer set to `seq_len` - `reuse_len` - 3. Constant across all calls.   
"""
def _split_a_and_b(data, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    if begin_idx + tot_len >= data_len:
        print("[_split_a_and_b] returns None: "
                        "begin_idx %d + tot_len %d >= data_len %d",
                        begin_idx, tot_len, data_len)
        return None

    # `end_idx` is now equal to `begin_idx` + `tot_len`.
    end_idx = begin_idx + tot_len

    a_begin = begin_idx
    label = 0
    a_end = end_idx

    b_len = 1
    # (zihang): `data_len - 1` to account for extend_target
    b_begin = random.randint(0, data_len - 1 - b_len)
    b_end = b_begin + b_len

    if DEBUG:
        """
        print("Initial b_begin:", b_begin)
        print("Initial b_end:", b_end)
        print("Initial b:", data[b_begin:b_end])
        """
        print("Initial b range:", "[", b_begin, ",", b_end, "]")
    b_begin = 0
    # (zihang): `data_len - 1` to account for extend_target
    b_end = data_len

    if DEBUG:
        """
        print("b after cut search:", data[b_begin:b_end])
        print("b_begin after cut search:", b_begin)
        print("b_end after cut search:", b_end)
        """
        print("Post cut_search b range:", "[", b_begin, ",", b_end, "]")
        print("============================")

    new_begin = a_end

    if DEBUG:
        # print("Initial a:", data[a_begin:a_end])
        # print("Initial b:", data[b_begin:b_end])
        print("Initial a range:", "[", a_begin, ",", a_end, "]")
        print("Initial b range:", "[", b_begin, ",", b_end, "]")
    # Keeps a and b the same size +/- 1. 
    # Shrinks their total size (len(a) + len(b)) to at most tot_len.
    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            # delete the right side only for the LM objective
            a_end -= 1
        else:
            b_end -= 1
    
    if DEBUG:
        print("Post resize a range:", "[", a_begin, ",", a_end, "]")
        print("Post resize b range:", "[", b_begin, ",", b_end, "]")
        # print("a_end after resize:", a_end)
        # print("b_end after resize:", b_end)
        print("a after resize:", data[a_begin:a_end])
        print("b after resize:", data[b_begin:b_end])
        print("============================")

    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        if a_end >= data_len or b_end >= data_len:
            print("[_split_a_and_b] returns None: "
                            "a_end %d or b_end %d >= data_len %d",
                            a_end, b_end, data_len)
            return None
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])
        
        if DEBUG:
            print("a_target:", a_target)
            print("b_target:", b_target)
            print("data_len - 1 - b_len:", data_len - 1 - b_len)

    if DEBUG:
        print('ret', ret)
    return ret

if __name__ == "__main__":
    data_len = 60
    seq_len = 30
    reuse_len = 15
    tot_len = seq_len - reuse_len - 3
    
    if DEBUG:
        print("seq_len:", seq_len)
        print("reuse_len:", reuse_len)
        print("tot_len:", tot_len)
        print("len(a) + len(b) == tot_len ==", tot_len)

    data = torch.tensor([i for i in range(data_len)])
    i = 0
    while i + seq_len <= data_len: # Keep going as long as we have one more full sequence to process. 
        if DEBUG:
            print("================START=================")
            print("============================")
        _split_a_and_b(data, i + reuse_len, tot_len, True)
        if DEBUG:
            print("================END===================")
        i += reuse_len
