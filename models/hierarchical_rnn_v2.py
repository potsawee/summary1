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
        encoder_output_dict = self.encoder(input, u_len, w_len)
        dec_output = self.decoder(target, encoder_output_dict)
        return dec_output

    def decode_beamsearch(self, input, u_len, w_len, decode_dict):
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
        search_method    = decode_dict['search_method']
        batch_size       = decode_dict['batch_size']
        time_step        = decode_dict['time_step']
        vocab_size       = decode_dict['vocab_size']
        device           = decode_dict['device']
        start_token_id   = decode_dict['start_token_id']
        stop_token_id    = decode_dict['stop_token_id']
        alpha            = decode_dict['alpha']
        penalty_ug       = decode_dict['penalty_ug']
        # length_offset    = decode_dict['length_offset']
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

        finished_beams = [[] for _ in range(batch_size)]

        beam_ht = [self.decoder.init_h0(batch_size) for _ in range(k)]
        finish = False
        for t in range(time_step-1):
            if finish: break
            decoder_output_t_array = torch.zeros((batch_size, k*vocab_size))

            for i, beam in enumerate(beams):

                # inference decoding
                # decoder_output = self.decoder(beam[:,:t+1], s_output, s_len, logsoftmax=True)[:,-1,:]
                decoder_output, beam_ht[i] = self.decoder.forward_step(beam[:,t:t+1], beam_ht[i], s_output, s_len, logsoftmax=True)

                # check if there is STOP_TOKEN emitted in the previous time step already
                # i.e. if the input at this time step is STOP_TOKEN
                for n_idx in range(batch_size):
                    if beam[n_idx][t] == stop_token_id: # already stop
                        decoder_output[n_idx, :] = float('-inf')
                        decoder_output[n_idx, stop_token_id] = 0.0 # to ensure STOP_TOKEN will be picked again!

                decoder_output_t_array[:,i*vocab_size:(i+1)*vocab_size] = decoder_output

                # add previous beam score bias
                for n_idx in range(batch_size):
                    decoder_output_t_array[n_idx,i*vocab_size:(i+1)*vocab_size] += beam_scores[n_idx,i]

                    if search_method == 'argmax':
                        # Penalty term for repeated uni-gram
                        unigram_dict = {}
                        for tt in range(t+1):
                            v = beam[n_idx,tt].cpu().numpy().item()
                            if v not in unigram_dict: unigram_dict[v] = 1
                            else: unigram_dict[v] += 1
                        for vocab_id, vocab_count in unigram_dict.items():
                            decoder_output_t_array[n_idx,(i*vocab_size)+vocab_id] -= penalty_ug*vocab_count/(t+1)

                # only support batch_size = 1!
                if t == 0:
                    decoder_output_t_array[n_idx,(i+1)*vocab_size:] = float('-inf')
                    break

            if search_method == 'sampling':
                # Sampling
                scores  = np.zeros((batch_size, k))
                indices = np.zeros((batch_size, k))
                pmf = np.exp(decoder_output_t_array.cpu().numpy())
                for bi in range(batch_size):
                    if pmf[bi].sum() != 1.0:
                        pmf[bi] /= pmf[bi].sum()
                    sampled_ids = np.random.choice(k*vocab_size, size=k, p=pmf[bi])
                    for _s, s_id in enumerate(sampled_ids):
                        scores[bi, _s]  = decoder_output_t_array[bi, s_id]
                        indices[bi, _s] = s_id

            elif search_method == 'argmax':
                # Argmax
                topk_scores, topk_ids = torch.topk(decoder_output_t_array, k, dim=-1)
                scores = topk_scores.double().cpu().numpy()
                indices = topk_ids.double().cpu().numpy()

            new_beams = [torch.zeros((batch_size, time_step), dtype=torch.int64).to(device) for _ in range(k)]
            for r_idx, row in enumerate(indices):
                for c_idx, node in enumerate(row):
                    vocab_idx = node % vocab_size
                    beam_idx  = int(node / vocab_size)

                    new_beams[c_idx][r_idx,:t+1] = beams[beam_idx][r_idx,:t+1]
                    new_beams[c_idx][r_idx,t+1]  = vocab_idx

                    # if there is a beam that has [END_TOKEN] --- store it
                    if vocab_idx == stop_token_id:
                        finished_beams[r_idx].append(new_beams[c_idx][r_idx,:t+1+1])
                        scores[r_idx, c_idx] = float('-inf')

            # only support BATCH SIZE = 1
            count_stop = 0
            for ik in range(k):
                if scores[0,ik] == float('-inf'): count_stop += 1
            if count_stop == k: finish = True

            beams = new_beams
            if search_method == 'sampling':
                # normalisation the score
                scores = np.exp(scores)
                scores = scores / scores.sum(axis=-1).reshape(batch_size, 1)
                beam_scores = np.log(scores + 1e-20) # suppress warning log(zero)
            elif search_method == 'argmax':
                beam_scores = scores

            # print("=========================  t = {} =========================".format(t))
            # for ik in range(k):
            #     print("beam{}: [{:.5f}]".format(ik, scores[0,ik]),bert_tokenizer.decode(beams[ik][0].cpu().numpy()[:t+2]))
            # pdb.set_trace()

            if (t % 50) == 0:
                print("{}=".format(t), end="")
                sys.stdout.flush()
        print("{}=#".format(t))

        for bi in range(batch_size):
            if len(finished_beams[bi]) == 0:
                finished_beams[bi].append(beams[0][bi])

        summaries_id = [None for _ in range(batch_size)]
        # for j in range(batch_size): summaries_id[j] = beams[0][j].cpu().numpy()
        for j in range(batch_size):
            _scores = self.beam_scoring(finished_beams[j], s_output, s_len, alpha)
            summaries_id[j] = finished_beams[j][np.argmax(_scores)].cpu().numpy()
            print(bert_tokenizer.decode(summaries_id[j]))

        return summaries_id

    def beam_scoring(self, beams, s_output, s_len, alpha):
        # scores ===> same shape as the beams
        # beams  ===> batch_size = 1
        scores = [None for _ in range(len(beams))]
        for i, seq in enumerate(beams):
            timesteps = seq.size(0)
            decoder_output = self.decoder(seq.view(1, timesteps), s_output, s_len, logsoftmax=True)

            score = 0
            for t in range(timesteps-1):
                pred_t  = seq[t+1]
                score_t = decoder_output[0, t, pred_t]
                score  += score_t
            scores[i] = score.cpu().numpy().item() / (timesteps)**alpha
            # print("SCORING: beam{}: score={:2f} --- len={} --- norm_score={}".format(i,score.cpu().numpy().item(), timesteps, scores[i]))
            # pdb.set_trace()
        return scores


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
                                bias=True, batch_first=True, dropout=dropout, bidirectional=True)

        # utterance-level GRU layer (with  binary gate)
        self.adapt_gru_ulevel = AdaptiveGRU(input_size=2*self.rnn_hidden_size, hidden_size=self.rnn_hidden_size,
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
        self.gru_wlevel.flatten_parameters()
        w_output, _ = self.gru_wlevel(embed)
        w_len = w_len.reshape(-1)

        utt_input = torch.zeros((w_output.size(0), w_output.size(2)), dtype=torch.float).cuda()
        for idx, l in enumerate(w_len):
            utt_input[idx] = w_output[idx, l-1]

        # utterance-level GRU
        utt_input = utt_input.view(batch_size, num_utterances, 2*self.rnn_hidden_size)
        h0 = torch.zeros((1, self.rnn_hidden_size), dtype=torch.float).cuda()
        utt_output, segment_indices = self.adapt_gru_ulevel(utt_input, h0, u_len)

        # segment level
        s_len = [len(x) for x in segment_indices]
        max_num_segments = max(s_len)

        seg_input = torch.zeros((batch_size, max_num_segments, self.rnn_hidden_size), dtype=torch.float).cuda()
        for bn in range(batch_size):
            for s_idx, s_pos in enumerate(segment_indices[bn]):
                seg_input[bn, s_idx] = utt_output[bn, s_pos]

        self.gru_slevel.flatten_parameters()
        s_output, _ = self.gru_slevel(seg_input)

        # reshape the output at different levels
        # w_output => [batch_size, num_utt, num_words, 2*hidden]
        # u_output => [batch_size, num_utt, hidden]
        # s_output => [batch_size, num_seg, hidden]
        w_output = w_output.view(batch_size, num_utterances, num_words, -1)
        w_len    = w_len.view(batch_size, -1)
        w2_len   = [None for _ in range(batch_size)]
        for bn, _l in enumerate(u_len):
            w2_len[bn] = w_len[bn, :_l].sum().item()

        w2_output = torch.zeros((batch_size, max(w2_len), w_output.size(-1))).cuda()
        utt_indices = [[] for _ in range(batch_size)]
        for bn, l1 in enumerate(u_len):
            x = 0
            for j, l2 in enumerate(w_len[bn, :l1]):
                w2_output[bn, x:x+l2, :] = w_output[bn, j, :l2, :]
                x += l2.item()
                utt_indices[bn].append(x)

        encoder_output_dict = {
            's_output': s_output, 's_len': s_len,
            'u_output': utt_output, 'u_len': u_len, 'segment_indices': segment_indices,
            'w_output': w2_output, 'w_len': w2_len, 'utt_indices': utt_indices
        }

        return encoder_output_dict

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

        self.rnn = nn.GRU(embedding_dim, dec_hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.attention_s = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attention_u = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attention_w = nn.Linear(2*mem_hidden_size, dec_hidden_size)
        self.attn_softmax = nn.Softmax(dim=-1)


        self.output_layer = nn.Linear(dec_hidden_size+2*mem_hidden_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, target, encoder_output_dict, logsoftmax=True):
        s_output = encoder_output_dict['s_output']
        s_len    = encoder_output_dict['s_len']
        u_output = encoder_output_dict['u_output']
        u_len    = encoder_output_dict['u_len']
        w_output = encoder_output_dict['w_output']
        w_len    = encoder_output_dict['w_len']

        segment_indices = encoder_output_dict['segment_indices']
        utt_indices     = encoder_output_dict['utt_indices']


        embed = self.embedding(target)
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(embed)

        # attention mechanism LEVEL --- TopicSegment (s)
        scores_s = torch.bmm(rnn_output, self.attention_s(s_output).permute(0,2,1))
        for bn, l in enumerate(s_len):
            scores_s[bn,:,l:].fill_(float('-inf'))
        scores_s = self.attn_softmax(scores_s)

        # attention mechanism LEVEL --- Utterance (u)
        scores_u = torch.bmm(rnn_output, self.attention_u(u_output).permute(0,2,1))
        for bn, l in enumerate(u_len):
            scores_u[bn,:,l:].fill_(float('-inf'))
        scores_u = self.attn_softmax(scores_u)
        scores_su = torch.zeros(scores_u.shape).cuda()

        # attention mechanism LEVEL --- Word (w)
        scores_w = torch.bmm(rnn_output, self.attention_w(w_output).permute(0,2,1))
        for bn, l in enumerate(w_len):
            scores_w[bn,:,l:].fill_(float('-inf'))
        scores_w = self.attn_softmax(scores_w)
        scores_suw = torch.zeros(scores_w.shape).cuda()

        batch_size = target.size(0)
        # Segment -> Utterance
        for bn in range(batch_size):
            idx1 = 0
            idx2 = 0
            end_indices = segment_indices[bn]
            start_indices = [0] + [a+1 for a in end_indices[:-1]]
            for i in range(len(segment_indices[bn])):
                i1 = start_indices[i]
                i2 = end_indices[i]+1 # python
                scores_su[bn, :, i1:i2] = scores_s[bn, :, i].unsqueeze(-1)*scores_u[bn, :, i1:i2]

        # Utterance -> Word
        for bn in range(batch_size):
            idx1 = 0
            idx2 = 0
            end_indices = utt_indices[bn]
            start_indices = [0] + [a+1 for a in end_indices[:-1]]
            for i in range(len(utt_indices[bn])):
                i1 = start_indices[i]
                i2 = end_indices[i]+1 # python
                scores_suw[bn, :, i1:i2] = scores_su[bn, :, i].unsqueeze(-1)*scores_w[bn, :, i1:i2]

        scores_suw = self.attn_softmax(scores_suw)
        context_vec = torch.bmm(scores_suw, w_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output), dim=-1))

        if logsoftmax:
            dec_output = self.logsoftmax(dec_output)

        return dec_output

    def forward_step(self, xt, ht, enc_output, enc_output_len, logsoftmax=True):
        xt = self.embedding(xt) # xt => [batch_size, 1, input_size]
                                # ht => [batch_size, num_layers, hidden_size]
        ht1 = torch.zeros((ht.size(0), ht.size(1), ht.size(2))).cuda()
        for n in range(self.num_layers):
            rnn_layer_n = getattr(self, 'rnn_layer_{}'.format(n))
            _h = ht[:,n:n+1,:]
            if n == 0:
                _x = xt
                _, _h1 = rnn_layer_n(_x, _h) # h1 => [batch_size, 1, hidden_size]

            else:
                _x = self.dropout_layer(_h1) + _x # residual connection
                _, _h1 = rnn_layer_n(_x, _h) # h1 => [batch_size, 1, hidden_size]

            ht1[:,n:n+1,:] = _h1

        rnn_output = _h1

        # attention mechanism
        scores = torch.bmm(rnn_output, self.attention(enc_output).permute(0,2,1))
        # add bias -inf to the padded positions
        for bn, l in enumerate(enc_output_len):
            scores[bn,:,l:].fill_(float('-inf'))
        scores = self.attn_softmax(scores)
        context_vec = torch.bmm(scores, enc_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output, xt), dim=-1))

        if logsoftmax:
            dec_output = self.logsoftmax(dec_output)

        return dec_output[:,-1,:], ht1

    def init_h0(self, batch_size):
        h0 = torch.zeros((batch_size, self.num_layers, self.dec_hidden_size)).cuda()
        return h0
