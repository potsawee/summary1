import sys
import torch
import torch.nn as nn
import numpy as np
import pdb

from data_meeting import bert_tokenizer

class PointerGeneratorNetwork(nn.Module):
    def __init__(self, args, device):
        super(PointerGeneratorNetwork, self).__init__()
        self.device = device
        self.num_words_meeting = args['num_words_meeting']
        self.summary_length = args['summary_length']
        self.vocab_size = args['vocab_size']
        self.embedding_dim = args['embedding_dim']
        self.hidden_size = args['rnn_hidden_size']
        self.num_layers_enc = args['num_layers_enc']
        self.num_layers_dec = args['num_layers_dec']
        self.dropout = args['dropout']

        # Encoder
        self.encoder_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers_enc, bias=True,
                                   batch_first=True, dropout=self.dropout, bidirectional=True)
        # Decoder
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.decoder_rnn = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layers_dec, bias=True,
                                   batch_first=True, dropout=self.dropout, bidirectional=False)
        # Attention Mechanism
        self.attention_h = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.attention_s = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.attention_v = nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_softmax = nn.Softmax(dim=-1)

        # Linear Output
        self.decoder_linear1 = nn.Linear(3*self.hidden_size, self.hidden_size, bias=True)
        self.decoder_linear2 = nn.Linear(self.hidden_size, self.vocab_size, bias=True)
        self.decoder_softmax = nn.Softmax(dim=-1)

        # Pointer - Generator Mechanism
        self.pointer_c = nn.Linear(2*self.hidden_size, 1, bias=True)
        self.pointer_s = nn.Linear(self.hidden_size, 1, bias=False)
        self.pointer_x = nn.Linear(self.embedding_dim, 1, bias=False)

        self.param_init()
        self.to(device)

    def param_init(self):
        # Initialisation
        # zero out the bias term
        # don't zero out LayerNorm term e.g. transformer_encoder.layers.0.norm1.weight
        for name, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_normal_(p)
            else:
                # if name[-4:] == 'bias': p.data.zero_()
                if 'bias' in name: nn.init.zeros_(p)

    def forward(self, input, u_len, w_len, target):
        # Encoder
        uw_len, enc_output, input_pgn = self.forward_encoder(input, u_len, w_len)
        # Decoder
        dec_output = self.forward_decoder(uw_len, enc_output, target, input_pgn, training=True)
        return dec_output

    def forward_encoder(self, input, u_len, w_len):
        # Encoder
        # PGN - doesn't have hierarchical structure so
        #       reshape the input to [batch_size, num_words_meeting]
        batch_size = input.size(0)
        input_pgn = torch.zeros((batch_size, self.num_words_meeting),dtype=torch.long).to(self.device)
        uw_len = torch.zeros((batch_size), dtype=torch.long).to(self.device)

        # no [SEP] added between utterances! --- just concat all the utterances in a meeting dialogue
        for i in range(batch_size):
            wi = 0
            for j in range(u_len[i]):
                wl = w_len[i,j]
                if wi+wl > self.num_words_meeting: break
                input_pgn[i,wi:wi+wl] = input[i,j,:wl]
                wi += wl
            uw_len[i] = wi

        # Encoder Forward
        enc_embed = self.encoder_embedding(input_pgn)
        self.encoder_rnn.flatten_parameters()
        enc_output, _ = self.encoder_rnn(enc_embed) # enc_output => [batch_size, num_words_meeting, hidden_size*2]
        return uw_len, enc_output, input_pgn

    def forward_decoder(self, uw_len, enc_output, target, input_pgn, training=True):
        batch_size = target.size(0)

        # Decoder Forward
        dec_embed = self.decoder_embedding(target)
        self.decoder_rnn.flatten_parameters()
        dec_output, _ = self.decoder_rnn(dec_embed)

        # Attention Mechanism --- to get context vector
        Wh_ht = self.attention_h(enc_output)
        Ws_st = self.attention_s(dec_output)
        enc_num = self.num_words_meeting
        dec_num = self.summary_length

        et = torch.zeros((batch_size, dec_num, enc_num, self.hidden_size)).cuda()
        for i in range(enc_num):
            et[:,:,i,:] = torch.tanh(Wh_ht[:,i,:].unsqueeze(1) + Ws_st[:,:,:])
        et = self.attention_v(et).squeeze(-1)
        et_mask = torch.zeros((batch_size, enc_num)).cuda()
        for bi, l in enumerate(uw_len): et_mask[bi,l:].fill_(float('-inf'))
        et = et + et_mask.unsqueeze(1)
        at = self.attention_softmax(et) # [batch_size, dec_num, enc_num]
        context_vec = torch.bmm(at, enc_output) # [batch_size, dec_num, hidden_size*2]

        # Output Layer
        output = self.decoder_linear1(torch.cat((dec_output,context_vec),dim=-1))
        output = self.decoder_softmax(self.decoder_linear2(output)) # [batch_size, dec_num, vocab_size]

        # # P(generating)
        Pc_ct = self.pointer_c(context_vec).squeeze(-1)
        Ps_st = self.pointer_s(dec_output).squeeze(-1)
        Px_xt = self.pointer_x(dec_embed).squeeze(-1)
        Pgen = torch.sigmoid(Pc_ct+Ps_st+Px_xt).unsqueeze(-1)

        # overall P(w)
        Pvocab = output
        # during training --- it's teacher forcing
        Ppt = torch.zeros((batch_size, dec_num, self.vocab_size)).cuda()
        if training:
            for bi in range(batch_size):
                for t in range(dec_num-1):
                    yt = target[bi,t+1]
                    pos = (input_pgn[bi,:] == yt).float().cuda()
                    pt_attn = at[bi,t,:]
                    Ppt[bi,t,yt] = (pos*pt_attn).sum()
        else:
            for bi in range(batch_size):
                for t in range(dec_num-1):
                    for k in range(self.vocab_size):
                        pos = (input_pgn[bi,:] == k).float().cuda()
                        pt_attn = at[bi,t,:]
                        Ppt[bi,t,k] = (pos*pt_attn).sum()

        Pw = Pgen*Pvocab + (1.0-Pgen)*Ppt + 1e-100
        return torch.log(Pw)

    def forward_decoder_step(self, uw_len, enc_output, xt, ht, ct, input_pgn, training=True):
        batch_size = xt.size(0)

        # Decoder Forward
        dec_embed = self.decoder_embedding(xt)
        self.decoder_rnn.flatten_parameters()

        dec_output, (ht1, ct1) = self.decoder_rnn(dec_embed, (ht,ct))

        # Attention Mechanism --- to get context vector
        Wh_ht = self.attention_h(enc_output)
        Ws_st = self.attention_s(dec_output)
        enc_num = self.num_words_meeting
        dec_num = 1

        et = torch.zeros((batch_size, dec_num, enc_num, self.hidden_size)).cuda()
        for i in range(enc_num):
            et[:,:,i,:] = torch.tanh(Wh_ht[:,i,:].unsqueeze(1) + Ws_st[:,:,:])
        et = self.attention_v(et).squeeze(-1)
        et_mask = torch.zeros((batch_size, enc_num)).cuda()
        for bi, l in enumerate(uw_len): et_mask[bi,l:].fill_(float('-inf'))
        et = et + et_mask.unsqueeze(1)
        at = self.attention_softmax(et) # [batch_size, dec_num, enc_num]
        context_vec = torch.bmm(at, enc_output) # [batch_size, dec_num, hidden_size*2]

        # Output Layer
        output = self.decoder_linear1(torch.cat((dec_output,context_vec),dim=-1))
        output = self.decoder_softmax(self.decoder_linear2(output)) # [batch_size, dec_num, vocab_size]

        # # P(generating)
        Pc_ct = self.pointer_c(context_vec).squeeze(-1)
        Ps_st = self.pointer_s(dec_output).squeeze(-1)
        Px_xt = self.pointer_x(dec_embed).squeeze(-1)
        Pgen = torch.sigmoid(Pc_ct+Ps_st+Px_xt).unsqueeze(-1)

        # overall P(w)
        Pvocab = output
        # during training --- it's teacher forcing
        Ppt = torch.zeros((batch_size, dec_num, self.vocab_size)).cuda()
        if training:
            for bi in range(batch_size):
                for t in range(dec_num-1):
                    yt = target[bi,t+1]
                    pos = (input_pgn[bi,:] == yt).float().cuda()
                    pt_attn = at[bi,t,:]
                    Ppt[bi,t,yt] = (pos*pt_attn).sum()
        else:
            for bi in range(batch_size):
                for t in range(dec_num-1):
                    for k in range(self.vocab_size):
                        pos = (input_pgn[bi,:] == k).float().cuda()
                        pt_attn = at[bi,t,:]
                        Ppt[bi,t,k] = (pos*pt_attn).sum()

        Pw = Pgen*Pvocab + (1.0-Pgen)*Ppt + 1e-100
        return torch.log(Pw), (ht1, ct1)

    def decoder_init_h0c0(self, batch_size):
        h0 = torch.zeros((batch_size, self.num_layers_dec, self.hidden_size)).cuda()
        c0 = torch.zeros((batch_size, self.num_layers_dec, self.hidden_size)).cuda()
        return (h0, c0)

    def decode_beamsearch(self, input, u_len, w_len, decode_dict):
        k                = decode_dict['k']
        search_method    = decode_dict['search_method']
        batch_size       = decode_dict['batch_size']
        time_step        = decode_dict['time_step']
        vocab_size       = decode_dict['vocab_size']
        device           = decode_dict['device']
        start_token_id   = decode_dict['start_token_id']
        stop_token_id    = decode_dict['stop_token_id']
        alpha            = decode_dict['alpha']
        # length_offset    = decode_dict['length_offset']
        keypadmask_dtype = decode_dict['keypadmask_dtype']

        # create beam array & scores
        beams       = [None for _ in range(k)]
        beam_scores = np.zeros((batch_size, k))

        # we should only feed through the encoder just once!!
        uw_len, enc_output, input_pgn = self.forward_encoder(input, u_len, w_len) # memory

        # we run the decoder time_step times (auto-regressive)
        tgt_ids = torch.zeros((batch_size, time_step), dtype=torch.int64).to(device)
        tgt_ids[:,0] = start_token_id

        for i in range(k): beams[i] = tgt_ids

        finished_beams = [[] for _ in range(batch_size)]

        beam_htct = [self.decoder_init_h0c0(batch_size) for _ in range(k)]
        finish = False
        for t in range(time_step-1):
            if finish: break
            decoder_output_t_array = torch.zeros((batch_size, k*vocab_size))

            for i, beam in enumerate(beams):

                # inference decoding
                decoder_output, beam_htct[i] = self.forward_decoder_step(uw_len, enc_output, beam[:,t:t+1],
                                                                        beam_htct[i][0], beam_htct[i][1],
                                                                        input_pgn, training=False)

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



            print("=========================  t = {} =========================".format(t))
            for ik in range(k):
                print("beam{}: [{:.5f}]".format(ik, scores[0,ik]),bert_tokenizer.decode(beams[ik][0].cpu().numpy()[:t+2]))
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
            _scores = self.beam_scoring(finished_beams[j], enc_output, uw_len, input_pgn, alpha)
            summaries_id[j] = finished_beams[j][np.argmax(_scores)].cpu().numpy()
            print(bert_tokenizer.decode(summaries_id[j]))

        return summaries_id

    def beam_scoring(self, beams, enc_output, uw_len, input_pgn, alpha):
        # scores ===> same shape as the beams
        # beams  ===> batch_size = 1
        scores = [None for _ in range(len(beams))]
        for i, seq in enumerate(beams):
            timesteps = seq.size(0)
            decoder_output = self.forward_decoder(uw_len, enc_output, seq.view(1, timesteps), input_pgn, training=False)

            score = 0
            for t in range(timesteps-1):
                pred_t  = seq[t+1]
                score_t = decoder_output[0, t, pred_t]
                score  += score_t
            scores[i] = score.cpu().numpy().item() / (timesteps)**alpha
        return scores
