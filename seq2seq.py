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
num_embed = 40

encoder = mx.rnn.SequentialRNNCell()
decoder = mx.rnn.SequentialRNNCell()

for i in range(num_layers):
    encoder.add(mx.rnn.LSTMCell(num_hidden, prefix='encoder_%d_' % i))
    decoder.add(mx.rnn.LSTMCell(num_hidden, prefix='decoder_%d_' % i))

src_data = mx.sym.var('src_data')
target_data = mx.sym.var('target_data')
label = mx.sym.var('softmax_label')
attn_fc_weight = mx.sym.var('attn_fc_weight')
attn_fc_bias = mx.sym.var('attn_fc_bias')

src_embed = mx.sym.Embedding(data=src_data, input_dim=src_dim, 
                             output_dim=num_embed, name='src_embed') 
target_embed = mx.sym.Embedding(data=target_data, input_dim=target_dim, 
                                output_dim=num_embed, name='target_embed')
def sym_gen(seq_len):
    enc_len, dec_len = seq_len

    encoder.reset()
    for cell in encoder._cells:
    	cell.reset()

    encoder_outputs, encoder_state = encoder.unroll(enc_len, inputs=src_embed, merge_outputs=True)

    decoder.reset()
    for cell in decoder._cells:
    	cell.reset()

    inputs, _ = _normalize_sequence(dec_len, target_embed, 'NTC', False)

    state = encoder_state
    outputs = []

    last_context = mx.sym.zeros((batch_size, num_hidden), name='before_first_context')
    # TODO: refactor
    for i in range(dec_len):
        dec_input = mx.sym.concat(inputs[i], last_context, dim=1)
        # (b, n)
        decoder_output, state = decoder(dec_input, state)
        
        fc_decoder_output = mx.sym.FullyConnected(data=decoder_output, weight=attn_fc_weight, bias=attn_fc_bias,
                                                  num_hidden=num_hidden, name='fc_decoder_output_%d' % i)
        # (b, 1, n)  n is num_hidden
        fc_decoder_output = mx.sym.expand_dims(data=fc_decoder_output, axis=1,
                                               name='expand_fc_decoder_output_%d' % i)
        # (b, 1, n) (b, seq, n) => (b, 1, seq)
        attn_weight = mx.sym.batch_dot(fc_decoder_output, encoder_outputs, 
                                       transpose_b=True, name='attn_weight_%d' % i)
        attn_weight = mx.sym.softmax(data=attn_weight, name='normalized_attn_weight_%d' % i)
        # (b, 1, seq) (b, seq, n) => (b, 1, n)
        last_context = mx.sym.batch_dot(attn_weight, encoder_outputs, 
                                        name='context_%d' % i)
        # (b, n)
        last_context = mx.sym.reshape(data=last_context, shape=(0, -3), name='squeeze_context_%d' % i)

        output = mx.sym.concat(decoder_output, last_context, dim=1, name='output_%d' % i)

        outputs.append(output)


    outputs, _ = _normalize_sequence(dec_len, outputs, 'NTC', True)

    label_reshape = mx.sym.reshape(data=label, shape=(-1,), name='label_reshape')

    outputs = mx.sym.reshape(outputs, shape=(-3, -1), name='outputs_reshape')
    fc = mx.sym.FullyConnected(data=outputs, num_hidden=target_dim, name='fc')
    pred = mx.sym.SoftmaxOutput(data=fc, label=label_reshape, name='softmax')
    # reshape for acc metric
    pred_reshape = mx.sym.reshape(data=pred, shape=(-1, dec_len, target_dim), name='pred_reshape')

    return pred_reshape, ('src_data', 'target_data'), ('softmax_label',)


batch_size = 5
buckets = [(5, 10), (10, 20), (20, 40)]

train_iter = DummySeq2seqIter(batch_size, buckets, ['src_data', 'target_data'], ['softmax_label'],
                              src_dim, target_dim)

seq2seq_mod = mx.mod.BucketingModule( 
        sym_gen = sym_gen,
        default_bucket_key = train_iter.default_bucket_key,
        context = mx.cpu(0))

seq2seq_mod.fit(train_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric=mx.metric.Accuracy(axis=2),
        batch_end_callback = mx.callback.Speedometer(batch_size, 10), 
        num_epoch=8)
