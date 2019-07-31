import tensorflow as tf
USE_EOD = False

def _create_data(input_paths):
  # Load sentence-piece model

  input_shards = []
  total_line_cnt = 0
  for input_path in input_paths:
    input_data, sent_ids = [], []
    sent_id, line_cnt = True, 0
    for line in tf.gfile.Open(input_path):
      line_cnt += 1

      if not line.strip(): # If at End Of Document:
        if USE_EOD:
          sent_id = not sent_id # Sent `sent_id` to `False`. 
          cur_sent = [EOD_ID]   
        else:
          continue
      else: # Otherwise, the line is nonempty. 
        if FROM_RAW_TEXT: # Handles space chars, quotes, weird stuff in text. 
          cur_sent = preprocess_text(line.strip(), lower=FLAGS.uncased)
          cur_sent = encode_ids(sp, cur_sent) # Converts to token ids. 
        else:
          cur_sent = list(map(int, line.strip().split())) # Converts ids as strs to ints. 

      input_data.extend(cur_sent)
      sent_ids.extend([sent_id] * len(cur_sent))
      sent_id = not sent_id

    tf.logging.info("Finish with line %d", line_cnt)
    if line_cnt == 0:
      continue

    input_data = np.array(input_data, dtype=np.int64)   # Shape: (<# tokens in `input_path`>,)
    sent_ids = np.array(sent_ids, dtype=np.bool)        # Shape: (<# tokens in `input_path`>,)

    total_line_cnt += line_cnt
    input_shards.append((input_data, sent_ids))

    # `input_data` is a np.array of token ids as integers for one input_path. 
    # `sent_ids` is an np.array of booleans for one input_path. It alternates between `True` and `False`
    #       on line/sentence breaks, and when a document ends if we are using EOD.  

  tfrecord_dir = os.path.join(FLAGS.save_dir, "tfrecords")

  filenames, num_batch = [], 0
  
  # =======================INPUT SHARD SHUFFLE===========================
  # Randomly shuffle input shards (with a fixed but distinct random seed)
  np.random.seed(100 * FLAGS.task + FLAGS.pass_id)

  perm_indices = np.random.permutation(len(input_shards))
  tf.logging.info("Using perm indices %s for pass %d",
                  perm_indices.tolist(), FLAGS.pass_id)

  input_data_list, sent_ids_list = [], []
  prev_sent_id = None
  for perm_idx in perm_indices:
    input_data, sent_ids = input_shards[perm_idx]
    # make sure the `send_ids[0] == not prev_sent_id`
    if prev_sent_id is not None and sent_ids[0] == prev_sent_id:
      sent_ids = np.logical_not(sent_ids)

    # append to temporary list
    input_data_list.append(input_data)
    sent_ids_list.append(sent_ids)

    # update `prev_sent_id`
    prev_sent_id = sent_ids[-1]

  input_data = np.concatenate(input_data_list)
  sent_ids = np.concatenate(sent_ids_list)
  # =======================INPUT SHARD SHUFFLE===========================

  file_name, cur_num_batch = create_tfrecords(
      save_dir=tfrecord_dir,
      basename="basename",
      data=[input_data, sent_ids],
      bsz_per_host=FLAGS.bsz_per_host,
      seq_len=FLAGS.seq_len,
      bi_data=FLAGS.bi_data,
  )

  filenames.append(file_name)
  num_batch += cur_num_batch

  record_info = {
      "filenames": filenames,
      "num_batch": num_batch
  }

  return record_info

