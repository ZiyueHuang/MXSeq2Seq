from mxnet import symbol
from data import DummySeq2seqIter
import mxnet as mx
import numpy as np
import logging


logging.getLogger().setLevel(logging.INFO)


def _normalize_sequence(length, inputs, layout, merge, in_layout=None):
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. Please convert " \
                "to list with list(inputs) first or let unroll handle splitting."
            # len([(batch, embed), ..., (batch, embed)]) = seq_length
            inputs = list(symbol.split(inputs, axis=in_axis, num_outputs=length,
                                       squeeze_axis=1))
    else:
        assert length is None or len(inputs) == length
        if merge is True:
            inputs = [symbol.expand_dims(i, axis=axis) for i in inputs]
            # (batch, seq_length, embed)
            inputs = symbol.Concat(*inputs, dim=axis)
            in_axis = axis

    if isinstance(inputs, symbol.Symbol) and axis != in_axis:
        inputs = symbol.swapaxes(inputs, dim0=axis, dim1=in_axis)

    return inputs, axis


src_dim = 5
target_dim = 10

num_layers = 3
num_hidden = 100
num_embed = 50

encoder = mx.rnn.SequentialRNNCell()
decoder = mx.rnn.SequentialRNNCell()

for i in range(num_layers):
    encoder.add(mx.rnn.LSTMCell(num_hidden, prefix='encoder_%d_' % i))
    decoder.add(mx.rnn.LSTMCell(num_hidden, prefix='decoder_%d_' % i))


def sym_gen(seq_len):
    src_data = mx.sym.var('src_data')
    target_data = mx.sym.var('target_data')
    label = mx.sym.var('softmax_label')

    src_embed = mx.sym.Embedding(data=src_data, input_dim=src_dim, 
                                 output_dim=num_embed, name='src_embed') 
    target_embed = mx.sym.Embedding(data=target_data, input_dim=target_dim, 
                                    output_dim=num_embed, name='target_embed')

    enc_len, dec_len = seq_len

    encoder.reset()
    for cell in encoder._cells:
    	cell.reset()

    encoder_outputs, encoder_states = encoder.unroll(enc_len, inputs=src_embed)

    decoder.reset()
    for cell in decoder._cells:
    	cell.reset()

    inputs, _ = _normalize_sequence(dec_len, target_embed, 'NTC', False)

    states = encoder_states
    outputs = []
    # TODO: attention
    for i in range(dec_len):
        decoder_output, states = decoder(inputs[i], states)
        outputs.append(decoder_output)

    outputs, _ = _normalize_sequence(dec_len, outputs, 'NTC', True)

    label_reshape = mx.sym.reshape(data=label, shape=(-1,), name='label_reshape')

    outputs_reshape = mx.sym.reshape(outputs, shape=(-1, num_hidden), name='outputs_reshape')
    pred = mx.sym.FullyConnected(data=outputs_reshape, num_hidden=target_dim, name='fc')
    pred = mx.sym.SoftmaxOutput(data=pred, label=label_reshape, name='softmax')
    # reshape for acc metric
    pred_reshape = mx.sym.reshape(data=pred, shape=(-1, dec_len, target_dim), name='pred_reshape')

    return pred_reshape, ('src_data', 'target_data'), ('softmax_label',)


batch_size = 5
buckets = [(5, 10), (10, 20), (20, 40)]

train_iter = DummySeq2seqIter(batch_size, buckets, ['src_data', 'target_data'], ['softmax_label'],
                                     src_dim, target_dim)

seq2seq_mod = mx.mod.BucketingModule( 
        sym_gen            = sym_gen,
        default_bucket_key = train_iter.default_bucket_key,
        context            = mx.cpu(0))

seq2seq_mod.fit(train_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric=mx.metric.Accuracy(axis=2),
        batch_end_callback = mx.callback.Speedometer(batch_size, 10), 
        num_epoch=8)
