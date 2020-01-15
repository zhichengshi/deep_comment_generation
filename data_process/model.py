import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import random


class BatchTreeEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 encode_dim,
                 batch_size,
                 use_gpu,
                 pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(
            Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

            batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))).cuda())

        for c in range(len(children)):
            zeros = self.create_tensor(
                Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(
                    0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(
            self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(
            Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class Encoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 encode_dim,
                 decode_dim,
                 batch_size,
                 use_gpu=True,
                 pretrained_weight=None):
        super(Encoder, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim,
                                        self.encode_dim, self.batch_size,
                                        self.gpu, pretrained_weight)
        # gru
        self.bigru = nn.GRU(self.encode_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=True,
                            batch_first=True)
        # linear
        self.fc = nn.Linear(encode_dim * 2, decode_dim)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(
                    torch.zeros(self.num_layers * 2, self.batch_size,
                                self.hidden_dim).cuda())
                c0 = Variable(
                    torch.zeros(self.num_layers * 2, self.batch_size,
                                self.hidden_dim).cuda())
                return h0, c0
            return Variable(
                torch.zeros(self.num_layers * 2, self.batch_size,
                            self.hidden_dim)).cuda()
        else:
            return Variable(
                torch.zeros(self.num_layers * 2, self.batch_size,
                            self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        src_lens = [len(item) for item in x]
        max_len = max(src_lens)
        encodes = []
        for i in range(self.batch_size):
            for j in range(src_lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(src_lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += src_lens[i]
            if max_len - src_lens[i]:
                seq.append(self.get_zeros(max_len - src_lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        # encodes = [batch_size, max_len, encoder_dim])
        encodes = encodes.view(self.batch_size, max_len, -1)
        packed_encodes = nn.utils.rnn.pack_padded_sequence(encodes, src_lens)
        # hidden = [batch, num_layer*2, hidden_dim]
        packed_output, hidden = self.bigru(packed_encodes, self.hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output)  # outpus = [batch, max_len, hidden_dim * 2]

        # init decoder hidden is final hidden state of the forwards and backwards
        # hidden = [batch, decode_dim]
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[:, -2, :], hidden[:, -1, :]), dim=-1)))

        return outputs, hidden, src_lens


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_dim):
        super().__init__()

        self.attention = nn.Linear(((encoder_hidden_dim * 2) + decoder_dim),
                                   decoder_dim)
        self.v = nn.Parameter(torch.rand(decoder_dim))

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch, decoder_dim]
        # encoder_outputs = [batch, max_len, encoder_hidden_dim]
        batch_size = encoder_outputs.shape[0]
        max_len = encoder_outputs.shape[1]

        # repeat encoder hidden state max_len times
        # hidden = [batch, max_len, decoder_dim]
        hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)

        # energy = [batch, max_len, decoder_dim]
        energy = torch.tanh(
            self.attention(torch.cat((hidden, encoder_outputs), dim=2)))

        v = self.v.repeat(batch_size,
                          1).unsqueeze(1)  # v = [batch, 1, decoder_dim]

        attention = torch.bmm(v, energy).squeeze(
            1)  # attention = [batch, max_len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, embedding_dim, encode_dim, decode_dim, output_dim, dropout, pretrained_weight, attention):
        super().__init__()
        self.num_layers = 1
        self.output_dim = output_dim
        self.attention = attention
        self.decode_dim = decode_dim

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU((encode_dim * 2) + embedding_dim, self.decode_dim, batch_first=True)
        self.fc_out = nn.Linear((encode_dim * 2) + decode_dim + embedding_dim, output_dim)

        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.rnn, nn.LSTM):
                h0 = Variable(
                    torch.zeros(self.num_layers, self.batch_size,
                                self.decode_dim).cuda())
                c0 = Variable(
                    torch.zeros(self.num_layers, self.batch_size,
                                self.decode_dim).cuda())
                return h0, c0
            return Variable(
                torch.zeros(self.num_layers, self.batch_size,
                            self.decode_dim)).cuda()
        else:
            return Variable(
                torch.zeros(self.num_layers, self.batch_size,
                            self.decode_dim))

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch]
        # hidden = [batch, decoder_dim]
        # encoder_outputs = [batch, max_len, encoder_hidden_dim *2 ]

        input = input.unsqueeze(1)  # input = [batch,1]

        embeded = self.dropout(self.embedding(input))  # embed = [batch, 1, embedding_dim]

        a = self.attention(hidden, encoder_outputs, mask)  # a = [batch, max_len]

        a = a.unsqueeze(1)  # a = [batch, 1, max_len]

        weighted = torch.bmm(a, encoder_outputs)  # weighted = [batch, 1 , encoder_hidden_dim * 2]

        rnn_input = torch.cat((embeded, weighted), dim=2)  # rnn_input = [batch, 1 , embedding_dim + encode_dim * 2]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(1))  # output = [batch, 1, decode_dim], hidden = [batch, 1, decode_dim]

        assert (output == hidden).all()  # this also means that output = hidden

        embeded = embeded.sequeeze(1)  # embeded = [batch, embedding_dim]
        output = output.sequeeze(1)  # output = [batch, decode_dim]
        weighted = weighted.squeeze(1)  # weight = [batch, encoder_hidden_dim * 2]

        prediction = self.fc_out(torch.cat((embeded, output, weighted), dim=1))  # prediction = [batch, output_dim]

        return prediction, hidden.squeeze(1), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, use_gpu=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_gpu = use_gpu

    def forward(self, src, trg, teacher_forcing_ratio=0.75):
        def create_mask(src_lens):
            max_len = max(src_lens)
            mask = []
            mask = np.array(mask)

            for i in range(len(src_lens)):
                row = np.concatenate(
                    (np.ones(src_lens[i]), np.zeros(max_len - src_lens[i])))
                mask = np.concatenate((mask, row))

            mask = mask.reshape(len(src_lens), max_len)
            return mask

        # src = [batch, None]
        # trg = [batch, trg_len]
        batch_size = len(src)
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder output
        if self.use_gpu:
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).cuda()

        # encoder_outputs is all hidden states of the input sequence, back and forward
        # hidden is the final forward and backward hidden states, pass through a linear layer

        encoder_outputs, hidden, src_lens = self.encoder(src)  # encoder)_outputs = [batch,src_len,]

        # first input to decoder is the <sos> tokens
        input = trg[:, 0]  # input = [batch, 1]

        mask = create_mask(src_lens)  # mask = [batch, src_len]

        for t in range(1, trg_len):
            # insert input token embedding ,previous states,encoder outputs and mask
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing ,use actual next token as next input
            # if not ,use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
