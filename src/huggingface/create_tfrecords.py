import tensorflow as tf

#========1=========2=========3=========4=========5=========6=========7=========8=========9=========0

def batchify(data, bsz_per_host, sent_ids=None):
  num_step = len(data) // bsz_per_host  # Number of timesteps per batch. 
  data = data[:bsz_per_host * num_step] # Truncate the data so that it's evenly divisble by bsz. 
  data = data.reshape(bsz_per_host, num_step)
  if sent_ids is not None:
    sent_ids = sent_ids[:bsz_per_host * num_step]
    sent_ids = sent_ids.reshape(bsz_per_host, num_step)

  if sent_ids is not None:
    return data, sent_ids
  return data

#========1=========2=========3=========4=========5=========6=========7=========8=========9=========0

def create_tfrecords(save_dir, basename, data, bsz_per_host, seq_len,
                     bi_data, sp):
  data, sent_ids = data[0], data[1] # Both 1-dimensional, with shape `(orig_data_len,)`. 
  # len(data) == len(sent_ids) == orig_data_len
  # `sent_ids` is an np.array of booleans for one input_path. It alternates between `True` and `False`
  #       on line/sentence breaks, and when a document ends if we are using EOD.  

  num_core = FLAGS.num_core_per_host
  bsz_per_core = bsz_per_host // num_core

  # The `batchify` function makes `data` 2-dimensional.  
  if bi_data:
    assert bsz_per_host % (2 * FLAGS.num_core_per_host) == 0
    fwd_data, fwd_sent_ids = batchify(data, bsz_per_host // 2, sent_ids)

    fwd_data = fwd_data.reshape(num_core, 1, bsz_per_core // 2, -1)
    fwd_sent_ids = fwd_sent_ids.reshape(num_core, 1, bsz_per_core // 2, -1)

    bwd_data = fwd_data[:, :, :, ::-1]
    bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

    data = np.concatenate(
        [fwd_data, bwd_data], 1).reshape(bsz_per_host, -1)
    sent_ids = np.concatenate(
        [fwd_sent_ids, bwd_sent_ids], 1).reshape(bsz_per_host, -1)
  else:
    data, sent_ids = batchify(data, bsz_per_host, sent_ids)

  # `data` has shape `(batch_size, data_len)`
  # `sent_ids` has shape `(batch_size, data_len)`
  # `batched_data_len` = `orig_data_len` // `batch_size`.
  # SHAPE MAPPING: (orig_data_len,) --> (batch_size, orig_data_len)   

  tf.logging.info("Raw data shape %s.", data.shape)

  file_name = format_filename(
      prefix=basename,
      bsz_per_host=bsz_per_host,
      seq_len=seq_len,
      bi_data=bi_data,
      suffix="tfrecords",
      mask_alpha=FLAGS.mask_alpha,
      mask_beta=FLAGS.mask_beta,
      reuse_len=FLAGS.reuse_len,
      uncased=FLAGS.uncased,
      fixed_num_predict=FLAGS.num_predict
  )
  save_path = os.path.join(save_dir, file_name)
  record_writer = tf.python_io.TFRecordWriter(save_path)
  tf.logging.info("Start writing %s.", save_path)

  num_batch = 0
  reuse_len = FLAGS.reuse_len

  # [sep] x 2 + [cls]
  assert reuse_len < seq_len - 3

  data_len = data.shape[1]
  sep_array = np.array([SEP_ID], dtype=np.int64)
  cls_array = np.array([CLS_ID], dtype=np.int64)

  i = 0
  while i + seq_len <= data_len: # Keep going as long as we have one more full sequence to process. 
    if num_batch % 500 == 0:
      tf.logging.info("Processing batch %d", num_batch)

    all_ok = True
    features = []
    for idx in range(bsz_per_host):     # For each row in batch: 
      inp = data[idx, i: i + reuse_len] # Grab a slice starting at `i` of size `reuse_len` 
      tgt = data[idx, i + 1: i + reuse_len + 1] # Grab a slice starting at `i+1` of size `reuse_len`

      results = _split_a_and_b(
          data[idx],
          sent_ids[idx],
          begin_idx=i + reuse_len, 
          tot_len=seq_len - reuse_len - 3,
          extend_target=True)
      if results is None:
        tf.logging.info("Break out with seq idx %d", i)
        all_ok = False
        break

      # unpack the results
      (a_data, b_data, label, _, a_target, b_target) = tuple(results)

      # sample ngram spans to predict
      reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1
      if FLAGS.num_predict is None:
        num_predict_0 = num_predict_1 = None
      else:
        num_predict_1 = FLAGS.num_predict // 2
        num_predict_0 = FLAGS.num_predict - num_predict_1
      mask_0 = _sample_mask(sp, inp, reverse=reverse,
                            goal_num_predict=num_predict_0)
      mask_1 = _sample_mask(sp, np.concatenate([a_data, sep_array, b_data,
                                                sep_array, cls_array]),
                            reverse=reverse, goal_num_predict=num_predict_1)

      # concatenate data
      cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                 sep_array, cls_array])
      seg_id = ([0] * (reuse_len + a_data.shape[0]) + [0] +
                [1] * b_data.shape[0] + [1] + [2])
      assert cat_data.shape[0] == seq_len
      assert mask_0.shape[0] == seq_len // 2
      assert mask_1.shape[0] == seq_len // 2

      # the last two CLS's are not used, just for padding purposes
      tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
      assert tgt.shape[0] == seq_len

      is_masked = np.concatenate([mask_0, mask_1], 0)
      if FLAGS.num_predict is not None:
        assert np.sum(is_masked) == FLAGS.num_predict

      feature = {
          "input": _int64_feature(cat_data),
          "is_masked": _int64_feature(is_masked),
          "target": _int64_feature(tgt),
          "seg_id": _int64_feature(seg_id),
          "label": _int64_feature([label]),
      }
      features.append(feature)

    if all_ok:
      assert len(features) == bsz_per_host
      for feature in features:
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        record_writer.write(example.SerializeToString())
      num_batch += 1
    else:
      break

    i += reuse_len

  record_writer.close()
  tf.logging.info("Done writing %s. Num of batches: %d", save_path, num_batch)

  return save_path, num_batch
