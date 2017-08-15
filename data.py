from random import choice
import mxnet as mx
import numpy as np


class DummySeq2seqIter(mx.io.DataIter):
    def __init__(self, batch_size, buckets,
                 data_names, label_names, src_dim, target_dim, dtype='float32',
                 layout='NT', invalid_label=-1, num_batches=50):
        self.num_batches = num_batches
        self.batch_size = batch_size      
        self.cur_batch = 0
        self.layout = layout
        self.buckets = buckets
        self.src_dim = src_dim
        self.target_dim = target_dim
        self.dtype = dtype
        self.data_names = data_names
        self.label_names = label_names

        self.default_bucket_key = (max([i[0] for i in buckets]), max([i[1] for i in buckets]))

        self.provide_data = [mx.io.DataDesc(name=name, 
                             shape=(self.batch_size, self.default_bucket_key[ind]),
                             layout=self.layout) for name, ind in zip(data_names, [0, 1])]

        self.provide_label = [mx.io.DataDesc(name=label_names[0],
                              shape=(self.batch_size, self.default_bucket_key[1]),
                              layout=self.layout)]

    def reset(self):
        self.cur_batch = 0

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            bucket = choice(self.buckets)
            src = np.random.choice(self.src_dim, (self.batch_size, bucket[0]))
            target = np.random.choice(self.target_dim, (self.batch_size, bucket[1]))
            label = np.random.choice(self.target_dim, (self.batch_size, bucket[1]))
            return mx.io.DataBatch([mx.nd.array(src, dtype=self.dtype), mx.nd.array(target, dtype=self.dtype)],
                         [mx.nd.array(label, dtype=self.dtype)], pad=0,
                         bucket_key=bucket,
                         provide_data=[mx.io.DataDesc(name=name, shape=(self.batch_size, bucket[ind]),
                             layout=self.layout) for name, ind in zip(self.data_names, [0, 1])],
                         provide_label=[mx.io.DataDesc(name=self.label_names[0],
                              shape=(self.batch_size, bucket[1]),
                              layout=self.layout)])
        else:
            raise StopIteration
