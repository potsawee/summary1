import torch
import torch.nn as nn
import pdb

from models.neural import AdaptiveGRU

class EncoderDecoder(nn.Module):
    def __init__(self, args, device):
        super(EncoderDecoder, self).__init__()
        self.device = device

        # Encoder - Hierarchical GRU
        self.encoder = HierarchicalGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'])

        # Decoder - GRU with attention mechanism
        self.decoder = DecoderGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'], args['rnn_hidden_size'],
                                  num_layers=4, dropout=0.1)

        # Initialisation
        # zero out the bias term
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.encoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if name[-4:] == 'bias': nn.init.zeros_(p)

        for name, p in self.decoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if name[-4:] == 'bias': nn.init.zeros_(p)

        self.to(device)

    def forward(self, input, u_len, w_len, target):
        s_output, s_len = self.encoder(input, u_len, w_len)
        dec_output = self.decoder(target, s_output, s_len)
        return dec_output

class HierarchicalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size):
        super(HierarchicalGRU, self).__init__()
        self.vocab_size      = vocab_size
        self.embedding_dim   = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)

        # word-level GRU layer: word-embeddings -> utterance representation
        self.gru_wlevel = nn.GRU(input_size=self.embedding_dim, hidden_size=self.rnn_hidden_size, num_layers=2,
                                bias=True, batch_first=True, dropout=0.1, bidirectional=False)

        # utterance-level GRU layer (with  binary gate)
        self.adapt_gru_ulevel = AdaptiveGRU(input_size=self.rnn_hidden_size, hidden_size=self.rnn_hidden_size,
                                            num_layers=2, bias=True)

        # segment-level GRU layer
        self.gru_slevel = nn.GRU(input_size=self.rnn_hidden_size, hidden_size=self.rnn_hidden_size, num_layers=2,
                                bias=True, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, input, u_len, w_len):
        # input => [batch_size, num_utterances, num_words]
        # embed => [batch_size, num_utterances, num_words, embedding_dim]
        # embed => [batch_size*num_utterances,  num_words, embedding_dim]

        batch_size     = input.size(0)
        num_utterances = input.size(1)
        num_words      = input.size(2)

        embed = self.embedding(input)
        embed = embed.view(batch_size*num_utterances, num_words, self.embedding_dim)

        # word-level GRU
        w_output, _ = self.gru_wlevel(embed)
        w_len = w_len.reshape(-1)

        utt_input = torch.zeros((w_output.size(0), w_output.size(2)), dtype=torch.float).cuda()
        for idx, l in enumerate(w_len):
            utt_input[idx] = w_output[idx, l-1]

        # utterance-level GRU
        utt_input = utt_input.view(batch_size, num_utterances, self.rnn_hidden_size)
        h0 = torch.zeros((1, self.rnn_hidden_size), dtype=torch.float).cuda()
        utt_output, segment_indices = self.adapt_gru_ulevel(utt_input, h0, u_len)

        # segment level
        s_len = [len(x) for x in segment_indices]
        max_num_segments = max(s_len)

        seg_input = torch.zeros((batch_size, max_num_segments, self.rnn_hidden_size), dtype=torch.float).cuda()
        for bn in range(batch_size):
            for s_idx, s_pos in enumerate(segment_indices[bn]):
                seg_input[bn, s_idx] = utt_output[bn, s_pos]

        s_output, _ = self.gru_slevel(seg_input)

        return s_output, s_len

class DecoderGRU(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, vocab_size, embedding_dim, dec_hidden_size, mem_hidden_size, num_layers=4, dropout=0.1):
        super(DecoderGRU, self).__init__()
        self.vocab_size  = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(embedding_dim, dec_hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.attention = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(embedding_dim+dec_hidden_size+mem_hidden_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, target, enc_output, enc_output_len):
        embed = self.embedding(target)

        rnn_output, _ = self.rnn(embed)

        # attention mechanism
        scores = torch.bmm(rnn_output, self.attention(enc_output).permute(0,2,1))
        # add bias -inf to the padded positions
        for bn, l in enumerate(enc_output_len):
            scores[bn,:,l:].fill_(float('-inf'))
        scores = self.attn_softmax(scores)
        context_vec = torch.bmm(scores, enc_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output, embed), dim=-1))
        dec_output = self.logsoftmax(dec_output)

        return dec_output
