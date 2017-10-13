from __future__ import unicode_literals

import argparse
import random
import unicodedata
import string
import re

import numpy as np
import mxnet as mx
from io import open
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F



parser = argparse.ArgumentParser(description='MXNet Gluon Seq2Seq Example')

parser.add_argument('--num-iters', type=int, default=75000,
                    help='number of iterations to train (default: 75000)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--max-length', type=int, default=10,
                    help='max length of sentence (default: 10)')
parser.add_argument('--num-layers', type=int, default=1,
                    help='number of layers in encoder and decoder (default: 1)')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='number of hidden units in encoder and decoder(default: 256)')
parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                    help='teacher forcing ratio (default: 0.5)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=5000, metavar='N',
                    help='how many iterations to wait before logging training status')
parser.add_argument('--test', action='store_true', default=False,
                    help='test layer by layer')

opt = parser.parse_args()


SOS_token = 0
EOS_token = 1



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


if not opt.test:
    input_lang, output_lang, pairs = prepareData('eng', 'fra', opt.max_length, True)
    print random.choice(pairs)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = F.array(indexes)
    return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)



class AttnDecoderRNN(Block):
    def __init__(self, hidden_size, output_size, n_layers, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        with self.name_scope():
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)
            self.attn = nn.Dense(self.max_length, in_units=self.hidden_size * 2)
            self.attn_combine = nn.Dense(self.hidden_size, in_units=self.hidden_size * 2)
            if self.dropout_p > 0:
                self.dropout = nn.Dropout(self.dropout_p)
            self.gru = rnn.GRU(self.hidden_size, input_size=self.hidden_size)
            self.out = nn.Dense(self.output_size, in_units=self.hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        #input shape, (1,)
        embedded = self.embedding(input)
        if self.dropout_p > 0:
            embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(F.concat(embedded, hidden[0].flatten(), dim=1)))
        attn_applied = F.batch_dot(attn_weights.expand_dims(0),
                                 encoder_outputs.expand_dims(0))

        output = F.concat(embedded.flatten(), attn_applied.flatten(), dim=1)
        output = self.attn_combine(output).expand_dims(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = self.out(output)

        return output, hidden, attn_weights

    def initHidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]


class EncoderRNN(Block):
    def __init__(self, input_size, hidden_size, n_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        with self.name_scope():
            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = rnn.GRU(hidden_size, input_size=self.hidden_size)

    def forward(self, input, hidden):
        ##input shape, (seq,)
        output = self.embedding(input).swapaxes(0, 1)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, ctx):
        return [F.zeros((1, 1, self.hidden_size), ctx=ctx)]




def train(input_variable, target_variable, encoder, decoder, teacher_forcing_ratio,
          encoder_optimizer, decoder_optimizer, criterion, max_length, ctx):
    with autograd.record():
        loss = F.zeros((1,), ctx=ctx)

        encoder_hidden = encoder.initHidden(ctx)

        input_length = input_variable.shape[0]
        target_length = target_variable.shape[0]

        encoder_outputs, encoder_hidden = encoder(
                input_variable.expand_dims(0), encoder_hidden)

        if input_length < max_length:
            encoder_outputs = F.concat(encoder_outputs.flatten(),
                F.zeros((max_length - input_length, encoder.hidden_size), ctx=ctx), dim=0)
        else:
            encoder_outputs = encoder_outputs.flatten()



        decoder_input = F.array([SOS_token], ctx=ctx)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                loss = F.add(loss, criterion(decoder_output, target_variable[di]))
                print criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topi = decoder_output.argmax(axis=1)

                decoder_input = F.array([topi.asscalar()], ctx=ctx)

                loss = F.add(loss, criterion(decoder_output, target_variable[di]))

                if topi.asscalar() == EOS_token:
                    break

        loss.backward()

    encoder_optimizer.step(1)
    decoder_optimizer.step(1)

    return loss.asscalar()/target_length



def trainIters(encoder, decoder, ctx, opt):
    #start = time.time()
    #plot_losses = []
    print_every = opt.log_interval
    print_loss_total = 0  # Reset every print_every

    encoder.initialize(ctx=ctx)
    decoder.initialize(ctx=ctx)

    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'sgd', {'learning_rate': opt.lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'sgd', {'learning_rate': opt.lr})

    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(opt.num_iters)]

    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    for iter in range(1, opt.num_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0].as_in_context(ctx)
        target_variable = training_pair[1].as_in_context(ctx)

        loss = train(input_variable, target_variable, encoder, decoder, opt.teacher_forcing_ratio,
                     encoder_optimizer, decoder_optimizer, criterion, opt.max_length, ctx)

        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print print_loss_avg



if opt.cuda:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu()

if opt.test:
    encoder = EncoderRNN(5, 10, 1)
    encoder.initialize()

    for i in encoder.collect_params().values():
        i.data()[:] = 1.0

    input = F.array([[0]])

    hidden = encoder.initHidden(ctx=mx.cpu())

    o, h = encoder(input, hidden)

    print 'encoder'
    print '=========='
    print o.asnumpy()
    print h[0].asnumpy()
    print '=========='

    attn_decoder = AttnDecoderRNN(2, 5, 1, 10, 0.1)
    attn_decoder.initialize()

    for i in attn_decoder.collect_params().values():
        i.data()[:] = 1.0

    input = F.array([0])
    hidden = attn_decoder.initHidden(ctx=mx.cpu())

    o, h, a = attn_decoder(input, hidden, 0.5*F.ones((10, 2)))

    print 'attn_decoder'
    print '=========='
    print o.asnumpy()
    print h[0].asnumpy()
    print '=========='

    assert False

encoder = EncoderRNN(input_lang.n_words, opt.hidden_size, opt.num_layers)
attn_decoder = AttnDecoderRNN(opt.hidden_size, output_lang.n_words,
                               opt.num_layers, opt.max_length, dropout_p=0.1)

trainIters(encoder, attn_decoder, ctx, opt)
