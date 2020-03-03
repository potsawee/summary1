import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_meeting import bert_tokenizer

class EncoderDecoder(nn.Module):
    def __init__(self, args, device):
        super(EncoderDecoder, self).__init__()
        self.device = device


        # Encoder - Hierarchical GRU
        self.encoder = HierarchicalGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_enc'], dropout=args['dropout'], device=device)

        # Decoder - GRU with attention mechanism
        self.decoder = DecoderGRU(args['vocab_size'], args['embedding_dim'], args['rnn_hidden_size'], args['rnn_hidden_size'],
                                       num_layers=args['num_layers_dec'], dropout=args['dropout'], device=device)

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
        enc_output_dict = self.encoder(input, u_len, w_len)
        dec_output, attn_scores, u_attn_scores = self.decoder(target, enc_output_dict)

        # compute coverage
        cov_scores = self.attn2cov(attn_scores)
        return dec_output, enc_output_dict['u_output'], attn_scores, cov_scores, u_attn_scores

    def attn2cov(self, attn_scores):
        batch_size, dec_steps, enc_steps = attn_scores.size()
        cov_scores = torch.zeros((batch_size, dec_steps, enc_steps), dtype=attn_scores.dtype).to(self.device)
        for t in range(1, dec_steps):
            cov_scores[:, t, :] = attn_scores[:, :t, :].sum(dim=1)
        return cov_scores

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
        keypadmask_dtype = decode_dict['keypadmask_dtype']

        # create beam array & scores
        beams       = [None for _ in range(k)]
        beam_scores = np.zeros((batch_size, k))

        # we should only feed through the encoder just once!!
        enc_output_dict = self.encoder(input, u_len, w_len) # memory
        u_output = enc_output_dict['u_output']

        # we run the decoder time_step times (auto-regressive)
        tgt_ids = torch.zeros((batch_size, time_step), dtype=torch.int64).to(device)
        tgt_ids[:,0] = start_token_id

        for i in range(k): beams[i] = tgt_ids

        finished_beams = [[] for _ in range(batch_size)]


        # initial hidden state
        ht = torch.zeros((self.decoder.num_layers, batch_size, self.decoder.dec_hidden_size),
                                    dtype=torch.float).to(self.device)
        for bn, l in enumerate(u_len): ht[:,bn,:] = u_output[bn,l-1,:].unsqueeze(0)
        beam_ht = [None for _ in range(k)]
        for _k in range(k): beam_ht[_k] = ht.clone()

        finish = False

        # attn_scores_array = None

        for t in range(time_step-1):
            if finish: break
            decoder_output_t_array = torch.zeros((batch_size, k*vocab_size))

            for i, beam in enumerate(beams):

                # inference decoding
                decoder_output, beam_ht[i], attn_scores = self.decoder.forward_step(beam[:,t:t+1], beam_ht[i], enc_output_dict, logsoftmax=True)

                # if attn_scores_array == None:
                #     enc_pos = attn_scores.size(-1)
                #     attn_scores_array = torch.zeros((k, time_step, enc_pos)) # BATCH_SIZE must be 1
                #
                # attn_scores_array[i,t,:] = attn_scores[0,0,:]

                # print("t = {}: attn_scores = {}".format(t , attn_scores))
                # import pdb; pdb.set_trace()

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
                # print("beam{}: [{:.5f}]".format(ik, scores[0,ik]),bert_tokenizer.decode(beams[ik][0].cpu().numpy()[:t+2]))
            # import pdb; pdb.set_trace()

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
            _scores = self.beam_scoring(finished_beams[j], enc_output_dict, alpha)
            summaries_id[j] = finished_beams[j][np.argmax(_scores)].cpu().numpy()
            print(bert_tokenizer.decode(summaries_id[j]))

        return summaries_id

    def beam_scoring(self, beams, enc_output_dict, alpha):
        # scores ===> same shape as the beams
        # beams  ===> batch_size = 1
        scores = [None for _ in range(len(beams))]
        for i, seq in enumerate(beams):
            timesteps = seq.size(0)
            decoder_output, _ = self.decoder(seq.view(1, timesteps), enc_output_dict, logsoftmax=True)

            score = 0
            for t in range(timesteps-1):
                pred_t  = seq[t+1]
                score_t = decoder_output[0, t, pred_t]
                score  += score_t
            scores[i] = score.cpu().numpy().item() / (timesteps)**alpha
            # print("SCORING: beam{}: score={:2f} --- len={} --- norm_score={}".format(i,score.cpu().numpy().item(), timesteps, scores[i]))
        return scores

class HierarchicalGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_hidden_size, num_layers, dropout, device):
        super(HierarchicalGRU, self).__init__()
        self.device          = device
        self.vocab_size      = vocab_size
        self.embedding_dim   = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size

        # embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim=self.embedding_dim, padding_idx=0)

        # word-level GRU layer: word-embeddings -> utterance representation
        # divide by 2 becuase bi-directional
        self.gru_wlevel = nn.GRU(input_size=self.embedding_dim, hidden_size=int(self.rnn_hidden_size/2), num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=True)

        # utterance-level GRU layer (with  binary gate)
        self.gru_ulevel = nn.GRU(input_size=self.rnn_hidden_size, hidden_size=int(self.rnn_hidden_size/2), num_layers=num_layers,
                                bias=True, batch_first=True, dropout=dropout, bidirectional=True)

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

        # utterance-level GRU
        utt_input = torch.zeros((w_output.size(0), w_output.size(2)), dtype=torch.float).to(self.device)
        for idx, l in enumerate(w_len):
            utt_input[idx] = w_output[idx, l-1]
        utt_input = utt_input.view(batch_size, num_utterances, self.rnn_hidden_size)
        self.gru_ulevel.flatten_parameters()
        utt_output, _ = self.gru_ulevel(utt_input)

        # reshape the output at different levels
        # w_output => [batch_size, num_utt, num_words, 2*hidden]
        # u_output => [batch_size, num_utt, hidden]
        w_output = w_output.view(batch_size, num_utterances, num_words, -1)
        w_len    = w_len.view(batch_size, -1)
        w2_len   = [None for _ in range(batch_size)]
        for bn, _l in enumerate(u_len):
            w2_len[bn] = w_len[bn, :_l].sum().item()

        w2_output = torch.zeros((batch_size, max(w2_len), w_output.size(-1))).to(self.device)
        utt_indices = [[] for _ in range(batch_size)]
        for bn, l1 in enumerate(u_len):
            x = 0
            for j, l2 in enumerate(w_len[bn, :l1]):
                w2_output[bn, x:x+l2, :] = w_output[bn, j, :l2, :]
                x += l2.item()
                utt_indices[bn].append(x-1) # minus one!!

        encoder_output_dict = {
            'u_output': utt_output, 'u_len': u_len,
            'w_output': w2_output, 'w_len': w2_len, 'utt_indices': utt_indices
        }

        return encoder_output_dict

class DecoderGRU(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, vocab_size, embedding_dim, dec_hidden_size, mem_hidden_size, num_layers, dropout, device):
        super(DecoderGRU, self).__init__()
        self.device      = device
        self.vocab_size  = vocab_size
        self.dec_hidden_size = dec_hidden_size
        self.mem_hidden_size = mem_hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(embedding_dim, dec_hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.attention_u = nn.Linear(mem_hidden_size, dec_hidden_size)
        self.attention_w = nn.Linear(mem_hidden_size, dec_hidden_size)

        self.output_layer = nn.Linear(dec_hidden_size+mem_hidden_size, vocab_size, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # utterance attention memory
        self.mem_utt_d = nn.Linear(vocab_size, 1, bias=False)
        self.mem_utt_y = nn.Linear(embedding_dim, 1, bias=False)
        self.mem_utt_s = nn.Linear(dec_hidden_size, 1, bias=True)

    def forward(self, target, encoder_output_dict, logsoftmax=True):
        batch_size = target.size(0)
        u_output = encoder_output_dict['u_output']
        u_len    = encoder_output_dict['u_len']
        w_output    = encoder_output_dict['w_output']

        # initial hidden state
        initial_h = torch.zeros((self.num_layers, batch_size, self.dec_hidden_size), dtype=torch.float).to(self.device)
        for bn, l in enumerate(u_len):
            initial_h[:,bn,:] = u_output[bn,l-1,:].unsqueeze(0)
        ht = initial_h

        target_len = target.size(1)
        scores_u  = torch.zeros((batch_size, target_len, u_output.size(1)), dtype=torch.float).to(self.device)
        scores_uw = torch.zeros((batch_size, target_len, w_output.size(1)), dtype=torch.float).to(self.device)
        dec_output = torch.zeros((batch_size, target_len, self.vocab_size), dtype=torch.float).to(self.device)
        for t in range(target_len):
            if t == 0:
                zero_attn_u = torch.zeros((batch_size, 1, u_output.size(1)), dtype=torch.float).to(self.device)
                dt = torch.zeros((batch_size, self.vocab_size), dtype=torch.float).to(self.device)
                dt, ht, score_uw, score_u = self.forward_step(target[:,t], ht, dt, zero_attn_u, encoder_output_dict, logsoftmax=True)
            else:
                dt, ht, score_uw, score_u = self.forward_step(target[:,t], ht, dt, score_u, encoder_output_dict, logsoftmax=True)

            dec_output[:,t,:] = dt
            scores_uw[:,t,:] = score_uw[:,0,:]
            scores_u[:,t,:]  = score_u[:,0,:]

        return dec_output, scores_uw, scores_u

    def forward_step(self, yt, ht, d_prev, eu_prev, encoder_output_dict, logsoftmax=True):
        u_output = encoder_output_dict['u_output']
        u_len    = encoder_output_dict['u_len']
        w_output = encoder_output_dict['w_output']
        w_len    = encoder_output_dict['w_len']

        utt_indices     = encoder_output_dict['utt_indices']

        batch_size = yt.size(0)
        yt = self.embedding(yt) # yt => [batch_size, 1, input_size]
                                # ht => [batch_size, num_layers, hidden_size]
        yt = yt.unsqueeze(1)

        rnn_output, ht1  = self.rnn(yt, ht)

        # attention mechanism LEVEL --- Utterance (u)
        scores_u = torch.bmm(rnn_output, self.attention_u(u_output).permute(0,2,1))
        for bn, l in enumerate(u_len):
            scores_u[bn,:,l:].fill_(float('-inf'))
        scores_u = F.softmax(scores_u, dim=-1)
        # compute gamma
        gamma = self.mem_utt_d(d_prev) + self.mem_utt_y(yt) + self.mem_utt_s(rnn_output)
        gamma = torch.sigmoid(gamma)
        scores_u = (1-gamma)*scores_u + gamma*eu_prev
        scores_u = scores_u / scores_u.sum(dim=-1) # when t == 0 --- we need to normalise

        # attention mechanism LEVEL --- Word (w)
        scores_w = torch.bmm(rnn_output, self.attention_w(w_output).permute(0,2,1))
        for bn, l in enumerate(w_len):
            scores_w[bn,:,l:].fill_(float('-inf'))
        scores_uw = torch.zeros(scores_w.shape).to(self.device)

        # Utterance -> Word
        for bn in range(batch_size):
            idx1 = 0
            idx2 = 0
            end_indices = utt_indices[bn]
            start_indices = [0] + [a+1 for a in end_indices[:-1]]
            for i in range(len(utt_indices[bn])):
                i1 = start_indices[i]
                i2 = end_indices[i]+1 # python
                scores_uw[bn, :, i1:i2] = scores_u[bn, :, i].unsqueeze(-1) * F.softmax(scores_w[bn, :, i1:i2], dim=-1)

        context_vec = torch.bmm(scores_uw, w_output)

        dec_output = self.output_layer(torch.cat((context_vec, rnn_output), dim=-1))

        if logsoftmax:
            dec_output = self.logsoftmax(dec_output)

        return dec_output[:,-1,:], ht1, scores_uw, scores_u

    def init_h0(self, batch_size):
        # h0 = torch.zeros((batch_size, self.num_layers, self.dec_hidden_size)).to(self.device)
        # swap dim 0,1 on 21 January 2020
        h0 = torch.zeros((self.num_layers, batch_size, self.dec_hidden_size)).to(self.device)

        return h0

class DALabeller(nn.Module):
    def __init__(self, rnn_hidden_size, num_da_acts, device):
        super(DALabeller, self).__init__()
        self.linear = nn.Linear(rnn_hidden_size, num_da_acts, bias=True)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.to(device)

    def forward(self, utt_output):
        return self.logsoftmax(self.linear(utt_output))

class EXTLabeller(nn.Module):
    def __init__(self, rnn_hidden_size, device):
        super(EXTLabeller, self).__init__()
        self.linear = nn.Linear(rnn_hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, utt_output):
        return self.sigmoid(self.linear(utt_output))
