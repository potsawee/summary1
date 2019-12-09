import sys
import torch
import torch.nn as nn
import numpy as np
import pdb

from models.neural import AdaptiveGRU
from data_meeting import bert_tokenizer

class EncoderDecoder(nn.Module):
    def __init__(self, args, device):
        super(EncoderDecoder, self).__init__()
        self.device = device


        # Encoder - Hierarchical GRU
        self.encoder = HierarchicalGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_enc'], dropout=args['dropout'])

        # Decoder - GRU with attention mechanism
        self.decoder = DecoderGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_dec'], dropout=args['dropout'])

        self.param_init()

        self.to(device)

    def param_init(self):
        # Initialisation
        # zero out the bias term
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.encoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)
        for name, p in self.decoder.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)

    def forward(self, input, u_len, w_len, target):
        s_output, s_len = self.encoder(input, u_len, w_len)
        dec_output = self.decoder(target, s_output, s_len)
        return dec_output

    def decode_beamsearch(self, input, u_len, w_len, decode_dict, target):
        """
        this method is meant to be used at inference time
            input = input to the encoder
            u_len = utterance lengths
            w_len = word lengths
            decode_dict:
                - k                = beamwidth for beamsearch
                - batch_size       = batch_size
                - time_step        = max_summary_length
                - vocab_size       = 30522 for BERT
                - device           = cpu or cuda
                - start_token_id   = ID of the start token
                - stop_token_id    = ID of the stop token
                - alpha            = length normalisation
                - length_offset    = length offset
                - keypadmask_dtype = torch.bool
        """
        k                = decode_dict['k']
        batch_size       = decode_dict['batch_size']
        time_step        = decode_dict['time_step']
        vocab_size       = decode_dict['vocab_size']
        device           = decode_dict['device']
        start_token_id   = decode_dict['start_token_id']
        stop_token_id    = decode_dict['stop_token_id']
        alpha            = decode_dict['alpha']
        length_offset    = decode_dict['length_offset']
        keypadmask_dtype = decode_dict['keypadmask_dtype']

        # create beam array & scores
        beams       = [None for _ in range(k)]
        beam_scores = np.zeros((batch_size, k))

        # we should only feed through the encoder just once!!
        s_output, s_len = self.encoder(input, u_len, w_len) # memory

        # we run the decoder time_step times (auto-regressive)
        tgt_ids = torch.zeros((batch_size, time_step), dtype=torch.int64).to(device)
        tgt_ids[:,0] = start_token_id

        for i in range(k): beams[i] = tgt_ids

        for t in range(time_step-1):
            decoder_output_t_array = torch.zeros((batch_size, k*vocab_size))

            for i, beam in enumerate(beams):

                # inference decoding
                # decoder_output = self.decoder(beam[:,:t+1], s_output, s_len, logsoftmax=False)[:,-1,:]
                # traching forcing
                decoder_output = self.decoder(target[:,:t+1], s_output, s_len, logsoftmax=False)[:,-1,:]


                # check if there is STOP_TOKEN emitted in the previous time step already
                # i.e. if the input at this time step is STOP_TOKEN
                for n_idx in range(batch_size):
                    if beam[n_idx][t] == stop_token_id: # already stop
                        decoder_output[n_idx, :] = float('-inf')
                        decoder_output[n_idx, stop_token_id] = 0.0 # to ensure STOP_TOKEN will be picked again!

                    else: # need to update scores --- length norm
                        beam_scores[n_idx,i] *= (t-1+length_offset)**alpha
                        beam_scores[n_idx,i] /= (t+length_offset)**alpha

                # length_norm = 1/(length)^alpha ... alpha = 0.7
                decoder_output_t_array[:,i*vocab_size:(i+1)*vocab_size] = decoder_output/(t+length_offset)**alpha

                # add previous beam score bias
                for n_idx in range(batch_size):
                    decoder_output_t_array[n_idx,i*vocab_size:(i+1)*vocab_size] += beam_scores[n_idx,i]

                if t == 0: break # only fill once for the first time step

            scores, indices = torch.topk(decoder_output_t_array, k=k, dim=-1)
            new_beams = [torch.zeros((batch_size, time_step), dtype=torch.int64).to(device) for _ in range(k)]
            for r_idx, row in enumerate(indices):
                for c_idx, node in enumerate(row):
                    vocab_idx = node % vocab_size
                    beam_idx  = int(node / vocab_size)

                    new_beams[c_idx][r_idx,:t+1] = beams[beam_idx][r_idx,:t+1]
                    new_beams[c_idx][r_idx,t+1]  = vocab_idx

            beam_scores = scores.cpu().numpy()
            beams = new_beams

            print(scores)
            print(bert_tokenizer.decode(beams[0][0].cpu().numpy()[:t+1]))
            pdb.set_trace()

        #     if (t % 100) == 0:
        #         print("{}=".format(t), end="")
        #         sys.stdout.flush()
        #
        # print("{}=#".format(t))

        summaries_id = [None for _ in range(batch_size)]
        for j in range(batch_size): summaries_id[j] = beams[0][j].cpu().numpy()

        return summaries_id


class HierarchicalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, num_layers, dropout):
        super(HierarchicalGRU, self).__init__()
        self.vocab_size      = vocab_size
        self.embedding_dim   = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)

        # word-level GRU layer: word-embeddings -> utterance representation
        self.gru_wlevel = nn.GRU(input_size=self.embedding_dim, hidden_size=self.rnn_hidden_size, num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=False)

        # utterance-level GRU layer (with  binary gate)
        self.adapt_gru_ulevel = AdaptiveGRU(input_size=self.rnn_hidden_size, hidden_size=self.rnn_hidden_size,
                                            num_layers=num_layers, bias=True, batch_first=True, dropout=dropout)

        # segment-level GRU layer
        self.gru_slevel = nn.GRU(input_size=self.rnn_hidden_size, hidden_size=self.rnn_hidden_size, num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=False)

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

    def __init__(self, vocab_size, embedding_dim, dec_hidden_size, mem_hidden_size, num_layers, dropout):
        super(DecoderGRU, self).__init__()
        self.vocab_size  = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        for n in range(num_layers):
            if n == 0:
                setattr(self, 'rnn_layer_{}'.format(n), nn.GRU(embedding_dim, dec_hidden_size, 1, batch_first=True, dropout=0.0))
            else:
                setattr(self, 'rnn_layer_{}'.format(n), nn.GRU(dec_hidden_size, dec_hidden_size, 1, batch_first=True, dropout=0.0))

        # self.rnn = nn.GRU(embedding_dim, dec_hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout_layer = nn.Dropout(p=dropout)

        self.attention = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attn_softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(embedding_dim+dec_hidden_size+mem_hidden_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, target, enc_output, enc_output_len, logsoftmax=True):
        embed = self.embedding(target)

        # rnn_output, _ = self.rnn(embed)
        for n in range(self.num_layers):
            rnn_layer_n = getattr(self, 'rnn_layer_{}'.format(n))
            if n == 0:
                _x = embed
                rnn_output, _ = rnn_layer_n(_x)
            else:
                _x = self.dropout_layer(rnn_output) + _x # residual connection
                rnn_output, _ = rnn_layer_n(_x)


        # attention mechanism
        scores = torch.bmm(rnn_output, self.attention(enc_output).permute(0,2,1))
        # add bias -inf to the padded positions
        for bn, l in enumerate(enc_output_len):
            scores[bn,:,l:].fill_(float('-inf'))
        scores = self.attn_softmax(scores)
        context_vec = torch.bmm(scores, enc_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output, embed), dim=-1))

        if logsoftmax:
            dec_output = self.logsoftmax(dec_output)

        return dec_output
