from torch.utils.data import Dataset
import torch
import pandas as pd

DEBUG = False
SIN = True

class XLSpredDataset(Dataset):
    def __init__(self, 
                 corpus_path, 
                 seq_len, 
                 num_predict,
                 data_batch_size,
                 reuse_len,
                 encoding="utf-8", 
                 corpus_lines=None, 
                 on_memory=True):

        self.seq_len = seq_len
        self.num_predict = num_predict
        self.data_batch_size = data_batch_size
        self.reuse_len = reuse_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.sample_counter = 0

        if SIN:
            self.raw_data = pd.read_csv('sin.csv')
            self.tensor_data = torch.tensor(self.raw_data.iloc[:,:].values)
        else:
            # Load samples into memory from file.
            self.raw_data = pd.read_csv(corpus_path)

            # Add and adjust columns.
            self.raw_data["Average"] = (self.raw_data["High"] + self.raw_data["Low"])/2
            self.raw_data['Volume'] = self.raw_data['Volume'] + 0.000001 # Avoid NaNs
            self.raw_data["Average_ld"] = (np.log(self.raw_data['Average']) - 
                                        np.log(self.raw_data['Average']).shift(1))
            self.raw_data["Volume_ld"] = (np.log(self.raw_data['Volume']) - 
                                    np.log(self.raw_data['Volume']).shift(1))
            self.raw_data = self.raw_data[1:]

            # convert data to tensor of shape(rows, features)
            self.tensor_data = torch.tensor(self.raw_data.iloc[:,[7,8]].values)
        
        self.features = self.create_features(self.tensor_data)
        print('len of features:', len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


    def _batchify(self, data, batch_size):
        num_step = len(data) // batch_size
        data = data[:batch_size * num_step]
        data = data.reshape(batch_size, num_step)

        return data

    def create_features(self, tensor_data):
        """
        Returns a list of features of the form 
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = self.tensor_data.shape[0]
        seq_len = self.seq_len
        num_predict = self.num_predict
        batch_size = self.data_batch_size
        reuse_len = self.reuse_len

        if DEBUG:
            print('original_data_len', original_data_len)
            print('seq_len', seq_len)
            print('reuse_len', reuse_len)

        # batchify the tensor as done in original xlnet implementation
        # This splits our data into shape(batch_size, data_len)
        # NOTE: data holds indices--not raw data
        # TODO: Add ``bi_data`` block from ``data_utils.py``. 
        data = torch.tensor(self._batchify(np.arange(0, original_data_len), batch_size))
        data_len = data.shape[1]
        sep_array = torch.tensor(np.array([SEP_ID], dtype=np.int64))
        cls_array = torch.tensor(np.array([CLS_ID], dtype=np.int64))

        i = 0
        features = []
        while i + seq_len <= data_len:
            # TODO: Is ``all_ok`` supposed to be inside or outside outer loop?
            all_ok = True
            for idx in range(batch_size):
                inp = data[idx, i: i + reuse_len]
                tgt = data[idx, i + 1: i + reuse_len + 1]
                results = _split_a_and_b(
                    data[idx],
                    begin_idx=i + reuse_len,
                    tot_len=seq_len - reuse_len - 3,
                    extend_target=True)
                if results is None:
                    all_ok = False
                    break

                # unpack the results
                (a_data, b_data, label, _, a_target, b_target) = tuple(results)
                
                # sample ngram spans to predict
                # TODO: Add ``bi_data`` stuff above. 
                bi_data = False
                reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1

                # TODO: Pass in ``num_predict`` as an argument or class var?
                num_predict_1 = num_predict // 2
                num_predict_0 = num_predict - num_predict_1
               
                if DEBUG:
                    print("inp shape:", inp.shape)
                    print("num_predict_0:", num_predict_0) 
                mask_0 = _sample_mask(inp,
                                      reverse=reverse,
                                      goal_num_predict=num_predict_0)
                mask_1 = _sample_mask(torch.cat([a_data,
                                                 sep_array,
                                                 b_data,
                                                 sep_array,
                                                 cls_array]),
                                      reverse=reverse, 
                                      goal_num_predict=num_predict_1)

                # concatenate data
                cat_data = torch.cat([inp, a_data, sep_array, b_data,
                                            sep_array, cls_array])
                seg_id = torch.tensor([0] * (reuse_len + a_data.shape[0]) + [0] +
                                      [1] * b_data.shape[0] + [1] + [2])
                if DEBUG:
                    print("mask_0 shape:", mask_0.shape)
                
                # TODO: Should these even be here?
                assert cat_data.shape[0] == seq_len
                assert mask_0.shape[0] == seq_len // 2
                assert mask_1.shape[0] == seq_len // 2

                # the last two CLS's are not used, just for padding purposes
                tgt = torch.cat([tgt, a_target, b_target, cls_array, cls_array])
                assert tgt.shape[0] == seq_len

                mask_0 = torch.Tensor(mask_0)
                mask_1 = torch.Tensor(mask_1)
                is_masked = torch.cat([mask_0, mask_1], 0)
                if DEBUG:
                    print('cat_data', cat_data)
                    print('is_masked', is_masked)
                """
                We append a vector of NaNs to tensor_data to serve as our ``[SEP]``, ``[CLS]`` vector.
                So ``mod_tensor_data`` is just ``tensor_data`` with this NaN vector added to the end. 
                ``zeroed_cat_data`` only modifies indices less than zero, and changes them to point to
                the NaN vector in ``mod_tensor_data``. Thus ``input_raw`` is the raw data with functional
                token indices yielding the NaN vector.

                Changed to zeroes temporarily.  
                """
                dim = tensor_data.shape[-1]
                nan_tensor = torch.Tensor([[1] * dim]).double()
                mod_tensor_data = torch.cat([tensor_data, nan_tensor])
                    
                nan_index = len(mod_tensor_data) - 1 
                zeroed_cat_data = torch.Tensor([nan_index if index < 0 else index for index in cat_data]).long() 
                if DEBUG:
                    print("type of ``zeroed_cat_data``:", type(zeroed_cat_data))
                    print("type of ``zeroed_cat_data[0]``:", type(zeroed_cat_data[0]))
                    print("``zeroed_cat_data[0]``:", zeroed_cat_data[0])
                input_raw = mod_tensor_data[zeroed_cat_data]

                # Do the same as above for ``tgt``. 
                zeroed_tgt = torch.Tensor([nan_index if index < 0 else index for index in tgt]).long() 
                tgt_raw = mod_tensor_data[zeroed_tgt]
                
                features.append((cat_data, input_raw, tgt_raw, is_masked, tgt, seg_id, label))
                
            if not all_ok:
                break

            i += reuse_len
        
        return features
